"""Turning a streamed LLM reply into speech, one sentence at a time.

Pure logic with no audio devices or models, so it's tested with fakes; talk.py wires
in the real microphone, speaker, LLMClient and TTSClient.
"""

import re
import time
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass

# A sentence ends at । ? ! or . followed by whitespace, or at a line break (models often
# end a line without punctuation). Requiring the space keeps "7.30" and "3.5" whole.
# ponytail: abbreviations ("Dr. Rao") split early; fine for spoken replies.
_SENTENCE_END = re.compile(r"(?<=[।?!.])\s+|\n+")


def sentences(deltas: Iterable[str]) -> Generator[str, None, None]:
    """Regroup streamed text deltas into whole sentences, each yielded as soon as it ends."""
    buffer = ""
    for delta in deltas:
        buffer += delta
        *done, buffer = _SENTENCE_END.split(buffer)  # the last part may still be growing
        yield from (s.strip() for s in done if s.strip())
    if buffer.strip():
        yield buffer.strip()  # the reply ended without a closing space


@dataclass
class SpokenReply:
    text: str
    first_token_s: float | None  # LLM time to first token
    first_audio_s: float | None  # time until the first PCM chunk was handed to the speaker


def speak_reply(
    deltas: Iterable[str],
    speak: Callable[[str], Iterable[bytes]],
    play: Callable[[bytes], None],
) -> SpokenReply:
    """Stream deltas → sentences → speak → play, timing the first token and first audio.

    Sequential on purpose: TTS renders a sentence in ~0.05x real time, so the LLM barely
    waits. play() must not block for the length of the audio (talk.py queues it), or
    sentence 2 would only start rendering after sentence 1 finished playing.
    """
    start = time.perf_counter()
    pieces: list[str] = []
    first_token: float | None = None
    first_audio: float | None = None

    def tap() -> Generator[str, None, None]:  # keep the full text and the first-token time
        nonlocal first_token
        for delta in deltas:
            if first_token is None:
                first_token = time.perf_counter() - start
            pieces.append(delta)
            yield delta

    for sentence in sentences(tap()):
        for pcm in speak(sentence):
            if first_audio is None:
                first_audio = time.perf_counter() - start
            play(pcm)
    return SpokenReply("".join(pieces), first_token, first_audio)
