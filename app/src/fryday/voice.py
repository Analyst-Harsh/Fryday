"""Turning a streamed LLM reply into speech, one sentence at a time.

Pure logic with no audio devices or models, so it's tested with fakes; talk.py wires
in the real microphone, speaker, LLMClient and TTSClient.
"""

import re
import time
from collections.abc import AsyncIterable, Awaitable, Callable, Generator, Iterable
from dataclasses import dataclass
from typing import Protocol

# A sentence ends at । ? ! or . followed by whitespace, or at a line break (models often
# end a line without punctuation). Requiring the space keeps "7.30" and "3.5" whole.
# ponytail: abbreviations ("Dr. Rao") split early; fine for spoken replies.
_SENTENCE_END = re.compile(r"(?<=[।?!.])\s+|\n+")

# Short first chunk (HLD §2 step 6): waiting for the whole first sentence was 68% of
# first-token → first-audio, ~105 ms per Devanagari word (EXPERIMENTS 2026-10-21).
FIRST_CHUNK_MIN_WORDS = 3  # cut at a comma only after this many words
FIRST_CHUNK_MAX_WORDS = 6  # otherwise cut after this many
_COMPLETE_WORD = re.compile(r"\S+(?=\s)")  # a word with a space after it can't still grow
_NUMBER = re.compile(r"[\d₹]")


def _first_chunk_end(buffer: str) -> int | None:
    """Where to cut the first chunk early, or None to keep waiting."""
    prev = ""
    for n, match in enumerate(_COMPLETE_WORD.finditer(buffer), start=1):
        word = match.group()
        # Never cut next to a number: "₹2.5 | लाख" or "₹250 | रुपये" breaks the normaliser.
        near_number = bool(_NUMBER.search(word) or _NUMBER.search(prev))
        prev = word
        if near_number:
            continue
        if (n >= FIRST_CHUNK_MIN_WORDS and word[-1] in ",;:") or n >= FIRST_CHUNK_MAX_WORDS:
            return match.end()
    return None


def split_chunks(buffer: str, first: bool) -> tuple[list[str], str]:
    """The speakable chunks finished in buffer, and the rest (which may still grow).

    first: no chunk has been yielded yet, so cut it early (a clause or ~6 words) and
    TTS can start; later chunks are whole sentences, which sound more natural.
    """
    *done, buffer = _SENTENCE_END.split(buffer)
    if first and not done and (cut := _first_chunk_end(buffer)) is not None:
        done, buffer = [buffer[:cut]], buffer[cut:]
    return [s.strip() for s in done if s.strip()], buffer


def sentences(deltas: Iterable[str]) -> Generator[str, None, None]:
    """Regroup streamed text deltas into speakable chunks, each yielded as soon as it ends."""
    buffer = ""
    first = True
    for delta in deltas:
        chunks, buffer = split_chunks(buffer + delta, first)
        first = first and not chunks
        yield from chunks
    if buffer.strip():
        yield buffer.strip()  # the reply ended without a closing space


class Channel(Protocol):
    """Where a voice turn's output goes: the terminal (talk.Terminal) or a browser
    (connection.Connection). Called only from the event loop."""

    def play(self, pcm: bytes, turn: int) -> None:
        """Queue audio; never waits for it to be heard."""
        ...

    def allow(self, turn: int) -> None:
        """Play only this turn's audio from now on (barge-in)."""
        ...

    def played(self, turn: int) -> int:
        """PCM bytes of this turn the user has heard so far."""
        ...

    def event(self, kind: str, **fields: object) -> None:
        """Something to show: thinking, transcript, reply, interrupted, notice, done."""
        ...


# What the user sees for each notice code; the browser gets both the code and the text.
NOTICES = {
    "heard_nothing": "(heard nothing, try again)",
    "llm_down": "(Fryday couldn't reach the model. Is `make llm` running?)",
    "too_short": "(too short, try again)",
    "silent": "(silence: is the mic allowed? Browser site settings, or System Settings >"
    " Privacy > Microphone for the terminal)",
}


@dataclass
class SpokenReply:
    text: str
    first_token_s: float | None  # LLM time to first token
    first_audio_s: float | None  # time until the first PCM chunk was handed to the speaker


async def speak_reply(
    deltas: AsyncIterable[str],
    speak: Callable[[str], Awaitable[list[bytes]]],
    play: Callable[[bytes], None],
) -> SpokenReply:
    """Stream deltas → chunks → speak → play, timing the first token and first audio.

    Sequential on purpose: TTS renders a chunk in ~0.05x real time, and the LLM keeps
    generating into reply()'s queue meanwhile. speak() must render off the event loop
    (talk.py uses to_thread), and play() must only queue the audio, not wait for it.
    Cancelling the task that awaits this stops it at the next await (barge-in).
    """
    start = time.perf_counter()
    pieces: list[str] = []
    first_token: float | None = None
    first_audio: float | None = None
    buffer, first = "", True

    async def say(chunk: str) -> None:
        nonlocal first_audio
        for pcm in await speak(chunk):
            if first_audio is None:
                first_audio = time.perf_counter() - start
            play(pcm)

    async for delta in deltas:
        if first_token is None:
            first_token = time.perf_counter() - start
        pieces.append(delta)
        chunks, buffer = split_chunks(buffer + delta, first)
        first = first and not chunks
        for chunk in chunks:
            await say(chunk)
    if buffer.strip():
        await say(buffer.strip())  # the reply ended without a closing space
    return SpokenReply("".join(pieces), first_token, first_audio)


def heard_text(chunks: list[tuple[str, int]], played: int) -> str:
    """The part of a reply that was played: whole chunks, then the cut chunk's words in
    proportion to its bytes played.

    chunks: (text, PCM bytes) in play order. ponytail: assumes words are evenly spread
    through the audio; TTS word timestamps would place the cut exactly.
    """
    heard: list[str] = []
    for text, size in chunks:
        if played >= size:
            heard.append(text)
            played -= size
            continue
        words = text.split()
        heard += words[: len(words) * played // size]
        break
    return " ".join(heard)
