"""Contract tests for the TTS seam.

The real TTSClient runs unchanged; only the speech model is replaced by a fake that
"speaks" fixed float arrays, so these run in milliseconds with no weights.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
from fryday.config import Settings
from fryday.tts import TTSClient, to_pcm16


@dataclass
class FakeKokoro:
    """Plays mlx-audio's Kokoro: one result with an .audio array per chunk."""

    chunks: list[list[float]]
    sample_rate: int = 24_000
    calls: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    produced: int = 0  # chunks actually generated, to prove .close() stops the work

    def generate(self, text: str, *, voice: str, lang_code: str) -> Iterator[SimpleNamespace]:
        self.calls.append({"text": text, "voice": voice, "lang_code": lang_code})
        for chunk in self.chunks:
            self.produced += 1
            yield SimpleNamespace(audio=np.array(chunk, dtype=np.float32))


def pcm(data: bytes) -> list[int]:
    return np.frombuffer(data, dtype="<i2").tolist()


# --- to_pcm16: the owner's function -------------------------------------------------


def test_to_pcm16_scales_and_clips() -> None:
    # 1.2 and -1.5 are out of range: clip to the ends, never wrap around.
    out = to_pcm16(np.array([0.0, 0.5, 1.0, -1.0, 1.2, -1.5], dtype=np.float32))
    assert pcm(out) == [0, 16384, 32767, -32767, 32767, -32767]


def test_to_pcm16_is_two_bytes_per_sample() -> None:
    assert len(to_pcm16(np.zeros(100, dtype=np.float32))) == 200


# --- TTSClient: the seam ------------------------------------------------------------


def test_speak_yields_one_pcm_chunk_per_model_chunk_in_order() -> None:
    fake = FakeKokoro(chunks=[[0.5], [1.0, -1.0]])
    tts = TTSClient(Settings(), model=fake)

    assert [pcm(c) for c in tts.speak("नमस्ते। कैसे हो?")] == [[16384], [32767, -32767]]
    assert tts.sample_rate == 24_000


def test_speak_passes_voice_and_language_from_settings() -> None:
    fake = FakeKokoro(chunks=[[0.0]])
    list(TTSClient(Settings(tts_voice="hm_omega", tts_lang="h"), model=fake).speak("नमस्ते"))

    assert fake.calls == [{"text": "नमस्ते", "voice": "hm_omega", "lang_code": "h"}]


def test_close_stops_generation_early() -> None:
    # Barge-in (v5): after the first chunk, the rest must never be rendered.
    fake = FakeKokoro(chunks=[[0.1], [0.2], [0.3]])
    speech = TTSClient(Settings(), model=fake).speak("एक। दो। तीन।")

    next(speech)
    speech.close()
    assert fake.produced == 1


def test_blank_text_yields_nothing_and_skips_the_model() -> None:
    fake = FakeKokoro(chunks=[[0.5]])
    tts = TTSClient(Settings(), model=fake)

    assert list(tts.speak("  \n ")) == []
    assert fake.calls == []
