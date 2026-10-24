"""One browser tab's WebSocket: protocol v1, the Connection channel and the socket readers.

Protocol v1 (HLD §2). Text frames are JSON {"v": 1, "t": <type>, ...}; binary frames are
PCM16 LE mono audio: 16 kHz up, ready.sample_rate down.
  up:   hello{token} (first, within 5 s) · played{turn, bytes} (on speech, see below)
  down: ready{sample_rate, turn, models} · speech{turn} (flush playback) · thinking ·
        transcript · reply (before its audio) · done{stats} · interrupted · notice{code, text}
Close codes: 1008 bad origin (before accept: HTTP 403) · 1013 not ready · 4400 bad message ·
4401 bad token · 4408 no hello · 4409 replaced by a newer tab.

No per-frame turn ids: one socket keeps order, and one sender drains one queue. Barge-in
calls allow(n+1) and then queues speech, so the browser flushes everything before
speech, and the sender drops old-turn audio queued after it.
"""

import asyncio
import json
import queue
import secrets
from typing import cast

import numpy as np
import structlog
from fastapi import WebSocket

from fryday import mic
from fryday.voice import NOTICES

log = structlog.stdlib.get_logger(__name__)

PROTOCOL = 1
HELLO_TIMEOUT_S = 5.0
FRAME_BYTES = mic.VAD_FRAME * 2  # PCM16: 2 bytes per sample


class _Nowhere:
    """The Channel while no browser is connected."""

    def play(self, pcm: bytes, turn: int) -> None: ...
    def allow(self, turn: int) -> None: ...
    def played(self, turn: int) -> int:  # noqa: ARG002
        return 0

    def event(self, kind: str, **fields: object) -> None: ...


NOWHERE = _Nowhere()


class Connection:
    """One browser tab as a voice.Channel. Only the event loop calls it."""

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self._out: asyncio.Queue[dict[str, object] | tuple[int, bytes]] = asyncio.Queue()
        self._turn = 0  # the turn whose audio may still be sent
        self._sent: dict[int, int] = {}  # audio bytes sent, per turn
        self._heard: dict[int, int] = {}  # audio bytes the browser says it played, per turn
        self.close_code: int | None = None  # set by stop(): why this tab is being closed
        self.receiver: asyncio.Task[None] | None = None
        self.closed = asyncio.Event()  # set once fully cleaned up (newest-wins waits on it)

    def play(self, pcm: bytes, turn: int) -> None:
        self._out.put_nowait((turn, pcm))

    def allow(self, turn: int) -> None:
        self._turn = turn

    def played(self, turn: int) -> int:
        # What the browser reported; if it never did (no barge-in, or the tab dropped
        # mid-reply), what was sent. Production gap: a drop keeps the whole reply.
        return self._heard.get(turn, self._sent.get(turn, 0))

    def event(self, kind: str, **fields: object) -> None:
        if kind == "notice":
            fields["text"] = NOTICES[str(fields["code"])]
        self._out.put_nowait({"v": PROTOCOL, "t": kind, **fields})

    def report_played(self, turn: int, n: int) -> None:
        # The max: a repeated or late report can only add heard audio, never erase it.
        self._heard[turn] = max(self._heard.get(turn, 0), n)

    def stop(self, code: int) -> None:
        """Close this tab with `code` (its handler's finally does the cleanup)."""
        self.close_code = code
        if self.receiver is not None:
            self.receiver.cancel()

    async def send_loop(self) -> None:
        while True:
            item = await self._out.get()
            if isinstance(item, dict):
                await self.ws.send_json(item)  # events always go: interrupted comes after allow
                continue
            turn, pcm = item
            if turn != self._turn:
                continue  # audio of a turn that was barged in on
            await self.ws.send_bytes(pcm)
            self._sent[turn] = self._sent.get(turn, 0) + len(pcm)


async def hello(ws: WebSocket, token: bytes) -> int | None:
    """The close code if the first message isn't a valid hello, else None."""
    try:
        message = await asyncio.wait_for(ws.receive(), HELLO_TIMEOUT_S)
    except TimeoutError:
        return 4408
    first = parse(message.get("text"))
    if first is None or first.get("t") != "hello":
        return 4400
    given = str(first.get("token", "")).encode()  # never logged
    return None if secrets.compare_digest(given, token) else 4401


async def receive(
    ws: WebSocket, conn: Connection, frames: "queue.Queue[np.ndarray | None]"
) -> None:
    """Read the socket until it closes: audio → 512-sample frames, played → conn."""
    pending = bytearray()
    dropped = 0
    while True:
        message = await ws.receive()
        if message["type"] == "websocket.disconnect":
            return
        if (data := message.get("bytes")) is not None:
            if len(cast(bytes, data)) % 2:
                conn.stop(4400)  # PCM16 is 2 bytes per sample
                return
            pending += cast(bytes, data)
            while len(pending) >= FRAME_BYTES:
                frame = np.frombuffer(bytes(pending[:FRAME_BYTES]), dtype="<i2")
                del pending[:FRAME_BYTES]
                try:
                    frames.put_nowait(frame)
                except queue.Full:  # the VAD fell behind: drop rather than grow
                    dropped += 1
                    if dropped % 100 == 1:
                        log.warning("upstream_audio_dropped", frames=dropped)
            continue
        msg = parse(message.get("text"))
        if msg is None:
            conn.stop(4400)
            return
        if msg.get("t") == "played":
            turn, n = msg.get("turn"), msg.get("bytes")
            if isinstance(turn, int) and isinstance(n, int):
                conn.report_played(turn, n)


def parse(text: object) -> dict[str, object] | None:
    """A protocol message, or None if it isn't JSON of our version with a type."""
    if not isinstance(text, str):
        return None
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(msg, dict):
        return None
    msg = cast(dict[str, object], msg)
    if msg.get("v") != PROTOCOL or not isinstance(msg.get("t"), str):
        return None
    return msg
