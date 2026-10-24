"""The WebSocket gateway with fake models and a fake VAD: handshake, turns, barge-in."""

import asyncio
import faulthandler
import json
import queue
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from fryday import connection, gateway
from fryday.config import Settings
from fryday.gateway import Models, create_app
from fryday.llm import TextDelta
from fryday.mic import VAD_FRAME, Vad
from pydantic import SecretStr
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

TOKEN = "test-token"
ORIGIN = {"origin": "http://localhost:5173"}
REPLY = "Kal dhoop rahegi. Shaam ko halki baarish hogi. "  # 2 chunks
LOUD = np.full(VAD_FRAME, 3000, dtype="<i2").tobytes()
QUIET = bytes(VAD_FRAME * 2)


@pytest.fixture(autouse=True)
def watchdog() -> Iterator[None]:
    # TestClient's receive() has no timeout: a wrong expectation would hang CI forever.
    faulthandler.dump_traceback_later(30, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


class FakeASR:
    def transcribe(self, _pcm: bytes, _rate: int) -> str:
        return "kal ka weather?"


class FakeLLM:
    def __init__(self, up: bool = True) -> None:
        self.up = up

    def ping(self) -> bool:
        return self.up

    def stream(self, _messages: object) -> Iterator[str]:  # the prefix-cache warm-up
        yield "ok"

    async def reply(self, _messages: object) -> AsyncIterator[TextDelta]:
        for word in REPLY.split(" "):
            await asyncio.sleep(0)
            yield TextDelta(word + " ")


class FakeTTS:
    sample_rate = 1000

    def __init__(self, second_chunk_s: float = 0.0) -> None:
        self.second_chunk_s = second_chunk_s  # time to barge in mid-reply

    def speak(self, text: str) -> Iterator[bytes]:
        if text.startswith("Shaam"):  # REPLY's 2nd sentence
            time.sleep(self.second_chunk_s)
        yield bytes(100)


def fake_silero() -> Callable[[], Vad]:
    """Per utterance: speech starts at the first loud frame, ends after 3 quiet ones."""

    def new() -> Vad:
        seen, speaking, quiet = 0, False, 0

        def vad(frame: np.ndarray) -> dict[str, int | float] | None:
            nonlocal seen, speaking, quiet
            seen += len(frame)
            loud = bool(np.abs(frame).max() > 1000)
            if not speaking:
                speaking = loud
                return {"start": seen - len(frame)} if loud else None
            quiet = 0 if loud else quiet + 1
            return {"end": seen} if quiet == 3 else None

        return vad

    return new


def make_app(
    tts: FakeTTS | None = None, llm: FakeLLM | None = None, loaded: threading.Event | None = None
) -> Any:
    def load(_settings: Settings) -> Models:
        if loaded is not None:
            loaded.wait(5)
        fakes: list[Any] = [FakeASR(), llm or FakeLLM(), tts or FakeTTS()]
        return Models(*fakes, new_vad=fake_silero)

    return create_app(Settings(access_token=SecretStr(TOKEN)), load=load)


def wait_ready(client: TestClient) -> None:
    for _ in range(200):
        if client.get("/readyz").status_code == 200:
            return
        time.sleep(0.01)
    raise AssertionError("never ready")


def hello(ws: WebSocketTestSession, token: str = TOKEN) -> dict[str, Any]:
    ws.send_text(json.dumps({"v": 1, "t": "hello", "token": token}))
    ready = ws.receive_json()
    assert ready["t"] == "ready"
    return ready


def say(ws: WebSocketTestSession) -> None:
    """An utterance: 0.48 s of speech, then the VAD's end of turn."""
    for frame in [LOUD] * 15 + [QUIET] * 4:
        ws.send_bytes(frame)


def until(ws: WebSocketTestSession, kind: str) -> list[str]:
    """Messages up to and including the first `kind` event; audio shows as 'audio'."""
    seen: list[str] = []
    while True:
        message = ws.receive()
        assert message["type"] == "websocket.send", message
        t = "audio" if message.get("bytes") is not None else json.loads(message["text"])["t"]
        seen.append(t)
        if t == kind:
            return seen


def test_readiness_waits_for_the_models_and_the_llm() -> None:
    loaded, llm = threading.Event(), FakeLLM(up=False)
    with TestClient(make_app(llm=llm, loaded=loaded)) as client:
        assert client.get("/healthz").json() == {"ok": True}
        assert client.get("/readyz").json() == {"ready": False, "reason": "warming up"}
        loaded.set()
        for _ in range(200):  # warm-up runs in the background
            if "unreachable" in client.get("/readyz").json().get("reason", ""):
                break
            time.sleep(0.01)
        assert client.get("/readyz").status_code == 503  # models warm, LLM down
        llm.up = True
        assert client.get("/readyz").json() == {"ready": True}


def test_a_websocket_from_another_origin_is_refused() -> None:
    with TestClient(make_app()) as client:
        wait_ready(client)
        with (
            pytest.raises(WebSocketDisconnect) as refused,
            client.websocket_connect("/ws", headers={"origin": "https://evil.example"}),
        ):
            pass
        assert refused.value.code == 1008


@pytest.mark.parametrize(
    ("first", "code"),
    [
        ({"v": 1, "t": "hello", "token": "wrong"}, 4401),
        ({"t": "hello", "token": TOKEN}, 4400),  # no protocol version
        ({"v": 1, "t": "played", "turn": 1, "bytes": 0}, 4400),  # not hello first
    ],
)
def test_the_first_message_must_be_a_valid_hello(first: dict[str, Any], code: int) -> None:
    with TestClient(make_app()) as client:
        wait_ready(client)
        with client.websocket_connect("/ws", headers=ORIGIN) as ws:
            ws.send_text(json.dumps(first))
            assert ws.receive() == {"type": "websocket.close", "code": code, "reason": ""}


def test_no_hello_in_time_closes_4408(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(connection, "HELLO_TIMEOUT_S", 0.05)
    with TestClient(make_app()) as client:
        wait_ready(client)
        with client.websocket_connect("/ws", headers=ORIGIN) as ws:
            assert ws.receive()["code"] == 4408


def test_a_turn_streams_events_and_audio_and_closing_cleans_up() -> None:
    app = make_app()
    with TestClient(app) as client:
        wait_ready(client)
        with client.websocket_connect("/ws", headers=ORIGIN) as ws:
            assert hello(ws)["sample_rate"] == 1000
            say(ws)
            seen = until(ws, "done")
            assert seen == [
                "speech",
                "thinking",
                "transcript",
                "reply",
                "audio",
                "reply",
                "audio",
                "done",
            ]
        state = app.state.fryday
        for _ in range(200):  # the handler's clean-up runs after the socket closes
            if state.current is None:
                break
            time.sleep(0.01)
        assert state.current is None
        assert state.session.out is connection.NOWHERE


def test_barge_in_flushes_audio_and_history_keeps_what_was_heard() -> None:
    app = make_app(tts=FakeTTS(second_chunk_s=0.5))
    with TestClient(app) as client:
        wait_ready(client)
        with client.websocket_connect("/ws", headers=ORIGIN) as ws:
            hello(ws)
            say(ws)
            until(ws, "audio")  # the 1st chunk is out; the 2nd renders for 0.5 s
            say(ws)  # the user talks over it
            seen = until(ws, "speech")
            # The browser played all 100 bytes of the 1st chunk (turn 2: VAD start
            # bumped it to 1, starting the turn to 2).
            ws.send_text(json.dumps({"v": 1, "t": "played", "turn": 2, "bytes": 100}))
            seen += until(ws, "done")

    after_speech = seen[seen.index("speech") :]
    assert after_speech.count("speech") == 1  # one speech per utterance, not per allow()
    assert "interrupted" in after_speech  # events still go out after allow(n+1)
    first_transcript = after_speech.index("transcript")
    assert "audio" not in after_speech[:first_transcript]  # nothing stale after the flush
    history = app.state.fryday.session._history
    assert history[2] == {"role": "assistant", "content": "Kal dhoop rahegi.…"}


def test_a_newer_tab_replaces_the_old_one_and_keeps_the_conversation() -> None:
    with TestClient(make_app()) as client:
        wait_ready(client)
        with client.websocket_connect("/ws", headers=ORIGIN) as first:
            hello(first)
            say(first)
            until(first, "done")
            with client.websocket_connect("/ws", headers=ORIGIN) as second:
                ready = hello(second)
                assert first.receive() == {"type": "websocket.close", "code": 4409, "reason": ""}
                assert ready["turn"] >= 2  # turn ids keep counting across tabs
                say(second)
                assert until(second, "done")[-1] == "done"


def test_the_close_marker_gets_through_a_full_queue() -> None:
    frames: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=2)
    frames.put(np.zeros(1))
    frames.put(np.zeros(1))
    gateway._put_last(frames, None)  # pyright: ignore[reportPrivateUsage]
    assert list(gateway._until_none(frames)) == [np.zeros(1)]  # pyright: ignore[reportPrivateUsage]
