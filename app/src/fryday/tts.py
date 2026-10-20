"""The TTS seam: text in, PCM16 audio out. The only module that touches the speech model.

Kokoro-82M via mlx-audio, in-process on the Mac today. A model server (Triton, v8)
can replace it behind the same speak() without touching callers.
"""

from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from fryday.config import Settings


class SpeechModel(Protocol):
    """The slice of mlx-audio's Kokoro model we use; tests pass a fake."""

    sample_rate: int

    def generate(self, text: str, *, voice: str, lang_code: str) -> Iterable[Any]: ...


def to_pcm16(audio: np.ndarray) -> bytes:
    """Float samples in [-1, 1] -> 16-bit little-endian PCM bytes (the wire format, HLD §2).

    Out-of-range samples are clipped, not wrapped: 1.2 must become 32767, not a loud click.
    """
    # Same ruler as F8: round (not truncate), so 0.5 -> 16384, not 16383.
    return np.round(np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def _load_kokoro(settings: Settings) -> tuple[SpeechModel, str]:
    """Load the pinned Kokoro snapshot: (model, path of the voice file)."""
    # Heavy imports only for the real model. hf_hub ships partial types.
    from huggingface_hub import snapshot_download  # pyright: ignore[reportUnknownVariableType]
    from mlx_audio.tts.utils import load_model

    # Pin model AND voice to one revision: by name, mlx-audio fetches voices
    # unpinned from another repo (EXPERIMENTS 2026-10-16).
    path = Path(
        snapshot_download(
            settings.tts_model,
            revision=settings.tts_revision,
            # Only the model and our voice: "*.safetensors" would pull all 56 voices.
            allow_patterns=[
                "*.json",
                "kokoro-v1_0.safetensors",
                f"voices/{settings.tts_voice}.safetensors",
            ],
        )
    )
    # load_model is typed as returning a bare nn.Module; Kokoro's has generate()/sample_rate.
    model = cast(SpeechModel, load_model(path))
    return model, str(path / "voices" / f"{settings.tts_voice}.safetensors")


class TTSClient:
    def __init__(self, settings: Settings, model: SpeechModel | None = None) -> None:
        # model: tests inject a fake here; None = load the pinned Kokoro snapshot.
        self._settings = settings
        if model is None:
            model, self._voice = _load_kokoro(settings)
        else:
            self._voice = settings.tts_voice
        self._model = model
        self.sample_rate: int = model.sample_rate

    def speak(self, text: str) -> Generator[bytes, None, None]:
        """Yield PCM16 mono chunks (about one per sentence) as they're generated.

        A generator, so the first chunk can play while later ones render, and
        callers can .close() it to stop speaking mid-reply (barge-in, v5).
        Expects normalised text: digits and ₹ are not handled here (v3 build 2).
        """
        if not text.strip():
            return
        for result in self._model.generate(
            text, voice=self._voice, lang_code=self._settings.tts_lang
        ):
            yield to_pcm16(np.asarray(result.audio))
