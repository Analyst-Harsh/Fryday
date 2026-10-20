"""v4 learn 1 toy: look inside Whisper with the owner's own recording.

1. Padding cost: does 3 s of audio take as long as 30 s?
2. The grid: the log-mel shape Whisper sees, before and after padding to 30 s.
3. Silence: what Whisper "hears" in pure silence and in quiet noise.

Run: uv run python spikes/learn/v4_asr_toy.py
"""

import statistics
import time
import wave

import numpy as np
from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim  # pyright: ignore[reportMissingTypeStubs]

from fryday.asr import SAMPLE_RATE, ASRClient
from fryday.config import Settings

CLIP = "data/audio/f8_16k.wav"  # the owner's 3 s F8 recording, 16 kHz mono


def load(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), "<i2")


def pcm(x: np.ndarray) -> bytes:
    return np.clip(x, -32768, 32767).astype("<i2").tobytes()


def timed(asr: ASRClient, audio: np.ndarray, runs: int = 3) -> tuple[float, str]:
    times, text = [], ""
    for _ in range(runs):
        t0 = time.perf_counter()
        text = asr.transcribe(pcm(audio), SAMPLE_RATE)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), text


def main() -> None:
    asr = ASRClient(Settings())
    voice = load(CLIP)
    asr.transcribe(pcm(voice), SAMPLE_RATE)  # warm-up: first call compiles kernels

    print("1. Padding cost (median of 3 runs)")
    for seconds in (3, 10, 30):
        audio = np.resize(voice, seconds * SAMPLE_RATE)  # repeat the clip to the length
        t, text = timed(asr, audio)
        print(f"   {seconds:>2} s audio → {t:.2f} s   ({len(text)} chars)")

    print("2. The grid")
    x = voice.astype(np.float32) / 32768.0
    mel = log_mel_spectrogram(x, n_mels=128)
    print(f"   {len(voice) / SAMPLE_RATE:.1f} s clip → log-mel {tuple(mel.shape)} (frames, mel bins)")
    print(f"   padded for the encoder → {tuple(pad_or_trim(mel, 3000, axis=-2).shape)}")

    print("3. Silence")
    rng = np.random.default_rng(0)
    for name, audio in [
        ("pure silence", np.zeros(5 * SAMPLE_RATE)),
        ("quiet noise ", rng.normal(0, 30, 5 * SAMPLE_RATE)),  # ~-60 dBFS, like a quiet room
    ]:
        print(f"   {name} → {asr.transcribe(pcm(audio), SAMPLE_RATE)!r}")


if __name__ == "__main__":
    main()
