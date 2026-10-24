"""The browser side of a voice chat: FastAPI serving the WebSocket gateway and the React app.

Run `make llm`, then `make serve` (and `make web` for the dev UI). Hands-free: the browser
streams its mic, and Silero VAD here ends each turn and catches barge-in, reusing the
terminal's turn logic (talk.VoiceSession, turns.TurnManager). The wire protocol and the
per-tab Connection live in connection.py.
"""

import asyncio
import contextlib
import queue
import secrets
import threading
import time
from collections.abc import AsyncGenerator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import structlog
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from silero_vad import load_silero_vad

from fryday import connection, mic
from fryday.asr import ASRClient
from fryday.config import Settings
from fryday.connection import NOWHERE, Connection
from fryday.llm import LLMClient
from fryday.logs import setup_logging
from fryday.talk import VoiceSession, warm_llm, warm_speech
from fryday.tracing import init_tracing, shutdown_tracing
from fryday.tts import TTSClient
from fryday.turns import TurnManager

log = structlog.stdlib.get_logger(__name__)

READY_CACHE_S = 5.0  # how long a good LLM ping counts, so /readyz polls stay cheap
UPSTREAM_FRAMES = 100  # ≈3.2 s of audio waiting for the VAD; more is dropped
WEB_DIST = Path("web/dist")  # ponytail: relative to the repo root, where `make serve` runs


@dataclass
class Models:
    asr: ASRClient
    llm: LLMClient
    tts: TTSClient
    # Per connection: a fresh Silero model (it keeps recurrent state, so tabs can't share
    # one) → the per-utterance VAD factory that mic.vad_loop wants.
    new_vad: Callable[[], Callable[[], mic.Vad]]


def load_models(settings: Settings) -> Models:
    def new_vad() -> Callable[[], mic.Vad]:
        # Silero ships untyped; the model is only handed back to its own VADIterator.
        return mic.silero(cast(object, load_silero_vad()), settings.vad_min_silence_ms)

    return Models(ASRClient(settings), LLMClient(settings), TTSClient(settings), new_vad)


@dataclass
class _State:
    models: Models | None = None
    session: VoiceSession | None = None
    turns: TurnManager | None = None
    current: Connection | None = None
    warm: bool = False
    failed: str | None = None  # why the models never loaded
    llm_ok_at: float = float("-inf")
    llm_warmed: bool = False
    background: set[asyncio.Task[Any]] = field(default_factory=lambda: set[asyncio.Task[Any]]())


def create_app(settings: Settings, load: Callable[[Settings], Models] = load_models) -> FastAPI:
    state = _State()
    token = settings.access_token.get_secret_value().encode()

    def spawn(coro: Any) -> None:  # keep a reference, or the loop may drop the task
        task = asyncio.get_running_loop().create_task(coro)
        state.background.add(task)
        task.add_done_callback(state.background.discard)

    async def boot() -> None:
        try:
            models = await asyncio.to_thread(load, settings)
            session = VoiceSession(
                settings,
                models.asr,
                models.llm,
                models.tts,
                NOWHERE,
                vad_wait_s=settings.vad_min_silence_ms / 1000,  # always hands-free here
            )
            # allow() follows whichever tab is connected now.
            turns = TurnManager(session.run_turn, lambda n: session.out.allow(n))
            state.models, state.session, state.turns = models, session, turns
            await asyncio.to_thread(warm_speech, models.asr, models.tts)
            state.warm = True
            log.info("gateway_warm")
        except Exception as e:
            state.failed = f"models failed to load: {e!r}"
            log.exception("gateway_boot_failed")

    async def not_ready() -> str | None:
        """Why turns can't run yet, or None. Pings the LLM at most every READY_CACHE_S."""
        if state.failed:
            return state.failed
        if not state.warm or state.models is None:
            return "warming up"
        if time.monotonic() - state.llm_ok_at < READY_CACHE_S:
            return None
        llm = state.models.llm
        if not await asyncio.to_thread(llm.ping):
            return "model server unreachable: is `make llm` running?"
        state.llm_ok_at = time.monotonic()
        if not state.llm_warmed:  # once: the first browser turn shouldn't pay the cold start
            state.llm_warmed = True
            spawn(asyncio.to_thread(_warm_quietly, settings, llm))
        return None

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        spawn(boot())  # serve /healthz and a 503 /readyz while the models load
        yield
        # uvicorn has already closed live sockets (1012); their handlers cleaned up.
        # Tracing is process-wide, so main() shuts it down, once, after uvicorn returns.
        for task in list(state.background):
            task.cancel()

    app = FastAPI(title="Fryday", lifespan=lifespan)
    app.state.fryday = state  # for tests and debugging

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:  # pyright: ignore[reportUnusedFunction]
        return {"ok": True}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        if reason := await not_ready():
            return JSONResponse({"ready": False, "reason": reason}, status_code=503)
        return JSONResponse({"ready": True})

    def barge_in(conn: Connection) -> None:
        """VAD heard speech start (on the loop). Also fires at the start of every turn."""
        if conn is not state.current or state.turns is None:
            return  # a late callback from a tab that has been replaced
        state.turns.interrupt()  # allow(n+1) first …
        conn.event("speech", turn=state.turns.turn_id)  # … then speech: order is the protocol

    def utterance(conn: Connection, pcm: bytes | None) -> None:
        if conn is not state.current or state.turns is None or pcm is None:
            return
        if code := mic.problem_with(pcm):
            conn.event("notice", code=code)
            return
        state.turns.start(pcm)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:  # pyright: ignore[reportUnusedFunction]
        if ws.headers.get("origin") not in settings.allowed_origins:
            await ws.close(code=1008)  # before accept: the browser sees HTTP 403
            return
        await ws.accept()
        if code := await connection.hello(ws, token):
            await ws.close(code=code)
            return
        session, turns, models = state.session, state.turns, state.models
        if await not_ready() or session is None or turns is None or models is None:
            await ws.close(code=1013)  # try again later: still loading, or no model server
            return

        conn = Connection(ws)
        if (old := state.current) is not None:  # newest wins
            old.stop(4409)
            await old.closed.wait()  # it settles history against its own `played` first
        state.current, session.out = conn, conn
        conn.event(
            "ready",
            sample_rate=models.tts.sample_rate,
            turn=turns.turn_id,
            models={
                "llm": settings.llm_model,
                "asr": settings.asr_model,
                "voice": settings.tts_voice,
            },
        )
        log.info("ws_connected")

        # The VAD thread must never touch asyncio objects directly: hand calls to the loop.
        loop = asyncio.get_running_loop()

        def on_speech() -> None:
            loop.call_soon_threadsafe(barge_in, conn)

        def on_utterance(pcm: bytes | None) -> None:
            loop.call_soon_threadsafe(utterance, conn, pcm)

        frames: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=UPSTREAM_FRAMES)
        new_vad = await asyncio.to_thread(models.new_vad)
        vad = threading.Thread(
            target=mic.vad_loop,
            args=(_until_none(frames), new_vad, on_speech, on_utterance),
            daemon=True,
        )
        vad.start()
        sender = asyncio.create_task(conn.send_loop())
        conn.receiver = asyncio.create_task(connection.receive(ws, conn, frames))

        async def clean_up() -> None:
            _put_last(frames, None)  # never dropped: the VAD thread must see it to exit
            await asyncio.to_thread(vad.join)
            if state.current is conn:
                if task := turns.interrupt():
                    await asyncio.gather(task, return_exceptions=True)
                session.settle()  # history keeps what this tab played, before re-pointing
                state.current, session.out = None, NOWHERE
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            if conn.close_code is not None:
                with contextlib.suppress(Exception):  # the socket may already be gone
                    await ws.close(code=conn.close_code)
            conn.closed.set()
            log.info("ws_closed", code=conn.close_code)

        try:
            await asyncio.gather(conn.receiver, return_exceptions=True)
        finally:
            # Shielded: if this handler is cancelled (server shutdown), clean-up still runs
            # to the end, or a newer tab would wait on `closed` forever.
            await asyncio.shield(clean_up())

    if WEB_DIST.is_dir():  # the built React app; after the routes, so /ws and /readyz win
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    return app


def _until_none(frames: "queue.Queue[np.ndarray | None]") -> Iterator[np.ndarray]:
    while (frame := frames.get()) is not None:
        yield frame


def _put_last(frames: "queue.Queue[np.ndarray | None]", item: None) -> None:
    """Put item even if the queue is full, by dropping the oldest audio frame."""
    while True:
        try:
            frames.put_nowait(item)
            return
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                frames.get_nowait()


def _warm_quietly(settings: Settings, llm: LLMClient) -> None:
    try:
        warm_llm(settings, llm)
    except Exception:  # only a speed-up; the ping already said the server is up
        log.warning("llm_warm_failed", exc_info=True)


def main() -> None:
    settings = Settings()
    setup_logging(settings.log_json)
    if not settings.access_token.get_secret_value():
        raise SystemExit(
            "Set FRYDAY_ACCESS_TOKEN (in .env) first, e.g.\n"
            f"  FRYDAY_ACCESS_TOKEN={secrets.token_urlsafe(32)}"
        )
    init_tracing(settings)
    print(f"Fryday on http://{settings.host}:{settings.port} (start `make llm` too)")
    try:
        uvicorn.run(
            create_app(settings),
            host=settings.host,
            port=settings.port,
            timeout_graceful_shutdown=5,
        )
    finally:
        shutdown_tracing()
