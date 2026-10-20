"""The ASR seam: PCM16 audio in, text out. The only module that touches the speech recogniser.

Whisper large-v3-turbo via mlx-whisper, in-process on the Mac today. A model server
(Triton, v8) can replace it behind the same transcribe() without touching callers.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np

from fryday.config import Settings

SAMPLE_RATE = 16_000  # Whisper's only input rate (F8: its log-mel front end is built for 16 kHz)

# mlx_whisper.transcribe's shape: (audio, *, path_or_hf_repo, language, ...) -> {"text": ...}
Transcribe = Callable[..., dict[str, Any]]


def _load_whisper(settings: Settings) -> tuple[Transcribe, str]:
    """The real transcribe function and the pinned snapshot's local path."""
    # Heavy imports only for the real model. hf_hub ships partial types.
    import mlx_whisper  # pyright: ignore[reportMissingImports]  # macOS-only; absent on Linux CI
    from huggingface_hub import snapshot_download  # pyright: ignore[reportUnknownVariableType]

    path = snapshot_download(settings.asr_model, revision=settings.asr_revision)
    fn = mlx_whisper.transcribe  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return cast(Transcribe, fn), str(Path(path))


class ASRClient:
    def __init__(self, settings: Settings, transcribe: Transcribe | None = None) -> None:
        # transcribe: tests inject a fake here; None = load the pinned Whisper snapshot.
        self._settings = settings
        if transcribe is None:
            self._transcribe, self._model = _load_whisper(settings)
        else:
            self._transcribe, self._model = transcribe, settings.asr_model

    def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        """Mono 16-bit little-endian PCM at 16 kHz -> the recognised text, stripped."""
        if sample_rate != SAMPLE_RATE:
            # Wrong-rate audio isn't an error Whisper reports: it just mishears (F8 quiz Q2).
            raise ValueError(f"ASR needs {SAMPLE_RATE} Hz audio, got {sample_rate} Hz")
        audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        # An array, not a file path: a path makes mlx-whisper shell out to ffmpeg.
        result = self._transcribe(
            audio,
            path_or_hf_repo=self._model,
            language=self._settings.asr_language,
            # Turns over 30 s: don't let one window's mistakes (or loops) leak into the next.
            condition_on_previous_text=False,
        )
        return str(result["text"]).strip()
