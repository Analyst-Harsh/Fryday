"""VoiceSession with fakes: its events, and after barge-in, history keeps what was heard."""

import asyncio
import sys
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fryday.config import Settings
from fryday.llm import TextDelta
from fryday.talk import Terminal, VoiceSession

REPLY = "Kal dhoop rahegi. Shaam ko halki baarish hogi. "  # 2 chunks, 100 bytes of audio each


class FakeASR:
    def transcribe(self, _pcm: bytes, _rate: int) -> str:
        return "kal ka weather?"


class FakeLLM:
    async def reply(self, _messages: object) -> AsyncIterator[TextDelta]:
        for word in REPLY.split(" "):
            yield TextDelta(word + " ")


class FakeTTS:
    def speak(self, _text: str) -> Iterator[bytes]:
        yield bytes(100)


class FakeChannel:
    def __init__(self, played: int = 0) -> None:
        self._played = played
        self.events: list[tuple[str, dict[str, object]]] = []

    def play(self, _pcm: bytes, turn: int) -> None: ...

    def allow(self, _turn: int) -> None: ...

    def played(self, _turn: int) -> int:
        return self._played

    def event(self, kind: str, **fields: object) -> None:
        self.events.append((kind, fields))


def test_a_turn_reports_its_steps_as_events() -> None:
    out = FakeChannel()
    fakes: list[Any] = [FakeASR(), FakeLLM(), FakeTTS(), out]
    asyncio.run(VoiceSession(Settings(), *fakes).run_turn(b"", 7))

    assert [kind for kind, _ in out.events] == ["thinking", "transcript", "reply", "reply", "done"]
    assert all(fields["turn"] == 7 for _, fields in out.events)
    assert [f["text"] for k, f in out.events if k == "reply"] == [
        "Kal dhoop rahegi.",
        "Shaam ko halki baarish hogi.",
    ]


@pytest.mark.parametrize(
    ("played", "kept"),
    [
        (200, REPLY),  # heard it all: untouched
        (150, "Kal dhoop rahegi. Shaam ko…"),  # cut halfway through the 2nd chunk
        (0, "…"),  # cut before any audio
    ],
)
def test_next_turn_trims_last_reply_to_what_was_heard(played: int, kept: str) -> None:
    fakes: list[Any] = [FakeASR(), FakeLLM(), FakeTTS(), FakeChannel(played)]
    session = VoiceSession(Settings(), *fakes)

    async def two_turns() -> None:
        await session.run_turn(b"", 1)
        await session.run_turn(b"", 2)  # the user spoke again: settles turn 1's reply

    asyncio.run(two_turns())
    history = cast(list[dict[str, str]], session._history)  # pyright: ignore[reportPrivateUsage]
    assert [m["role"] for m in history] == ["system", "user", "assistant", "user", "assistant"]
    assert history[2]["content"].strip() == kept.strip()


def test_terminal_prints_events_as_before(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Output:  # stands in for sd.RawOutputStream; no PortAudio needed
        latency = 0.02

        def __init__(self, **_kwargs: object) -> None: ...
        def start(self) -> None: ...
        def stop(self) -> None: ...
        def close(self) -> None: ...

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(RawOutputStream=Output))
    terminal = Terminal(sample_rate=1000)
    terminal.event("thinking", turn=1)
    terminal.event("transcript", turn=1, text="kal ka weather?")
    terminal.event("notice", turn=1, code="llm_down")
    stats = {"vad_wait_s": 0.5, "asr_s": 0.3, "llm_first_token_s": 0.1}
    terminal.event("done", turn=1, stats=stats | {"first_audio_s": 0.8, "heard_s": 1.6})
    terminal.close()

    assert capsys.readouterr().out.splitlines() == [
        "you>    kal ka weather?",
        "(Fryday couldn't reach the model. Is `make llm` running?)",
        "        ⏱  silence 500 ms → ASR 300 ms → first token +100 ms → first audio +800 ms"
        "  (you stop talking → Fryday speaks: 1600 ms)",
    ]
