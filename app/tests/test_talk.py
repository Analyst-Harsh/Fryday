"""VoiceSession with fakes: after barge-in, history keeps what was heard."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

import pytest
from fryday.config import Settings
from fryday.llm import TextDelta
from fryday.talk import VoiceSession

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


class FakeSpeaker:
    def __init__(self, played: int) -> None:
        self._played = played

    def play(self, _pcm: bytes, turn: int) -> None: ...

    def played(self, _turn: int) -> int:
        return self._played


@pytest.mark.parametrize(
    ("played", "kept"),
    [
        (200, REPLY),  # heard it all: untouched
        (150, "Kal dhoop rahegi. Shaam ko…"),  # cut halfway through the 2nd chunk
        (0, "…"),  # cut before any audio
    ],
)
def test_next_turn_trims_last_reply_to_what_was_heard(played: int, kept: str) -> None:
    fakes: list[Any] = [FakeASR(), FakeLLM(), FakeTTS(), FakeSpeaker(played)]
    session = VoiceSession(Settings(), *fakes)

    async def two_turns() -> None:
        await session.run_turn(b"", 1)
        await session.run_turn(b"", 2)  # the user spoke again: settles turn 1's reply

    asyncio.run(two_turns())
    history = cast(list[dict[str, str]], session._history)  # pyright: ignore[reportPrivateUsage]
    assert [m["role"] for m in history] == ["system", "user", "assistant", "user", "assistant"]
    assert history[2]["content"].strip() == kept.strip()
