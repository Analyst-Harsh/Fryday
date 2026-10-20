"""Contract tests for the ASR seam: the real ASRClient with a fake transcribe function."""

from typing import Any

import numpy as np
import pytest
from fryday.asr import ASRClient
from fryday.config import Settings


class FakeWhisper:
    """Plays mlx_whisper.transcribe: records what it was given, returns fixed text."""

    def __init__(self, text: str = " नमस्ते फ्राइडे ") -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def __call__(self, audio: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"audio": audio, **kwargs})
        return {"text": self.text, "language": "hi"}


def pcm16(*samples: int) -> bytes:
    return np.array(samples, dtype="<i2").tobytes()


def test_converts_pcm16_to_float_in_range() -> None:
    fake = FakeWhisper()
    ASRClient(Settings(), transcribe=fake).transcribe(pcm16(0, 16384, -32768, 32767), 16_000)

    audio = fake.calls[0]["audio"]
    assert audio.dtype == np.float32
    assert audio.tolist() == pytest.approx([0.0, 0.5, -1.0, 32767 / 32768])


def test_passes_model_and_language_and_strips_text() -> None:
    fake = FakeWhisper()
    text = ASRClient(Settings(asr_language="en"), transcribe=fake).transcribe(pcm16(0), 16_000)

    assert text == "नमस्ते फ्राइडे"
    call = fake.calls[0]
    assert call["language"] == "en"
    assert call["path_or_hf_repo"] == Settings().asr_model
    assert call["condition_on_previous_text"] is False  # each turn stands alone


def test_rejects_audio_that_is_not_16khz() -> None:
    # Whisper wouldn't complain about 48 kHz audio; it would just mishear it.
    fake = FakeWhisper()
    with pytest.raises(ValueError, match="16000 Hz"):
        ASRClient(Settings(), transcribe=fake).transcribe(pcm16(0), 48_000)
    assert fake.calls == []
