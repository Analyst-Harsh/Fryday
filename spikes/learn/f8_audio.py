# /// script
# requires-python = ">=3.12"
# dependencies = ["sounddevice", "soundfile", "librosa", "matplotlib", "numpy"]
# ///
"""F8 toy: look at your own voice the way the TTS/ASR models will. Throwaway learning code.

Records 3 s from the mic, prints the raw numbers, plots waveform / spectrogram / Whisper-style
log-mel, then plays it back at 48, 16 and 8 kHz. Audio and plot go to data/audio/ (gitignored).

Run from the repo root: uv run spikes/learn/f8_audio.py   (macOS asks once for mic permission)
"""

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf

SR, SECONDS = 48_000, 3  # 48 kHz is the Mac mic's native rate
OUT = Path("data/audio")
OUT.mkdir(parents=True, exist_ok=True)

# --- 1. record -------------------------------------------------------------------
input(f"Press Enter, then speak for {SECONDS} s, e.g. 'Fryday, kal subah 7 baje yaad dilana' ...")
x = sd.rec(SR * SECONDS, samplerate=SR, channels=1, dtype="float32")
sd.wait()
x = x[:, 0]  # (samples, 1 channel) -> (samples,)
print("...recorded.\n")

# --- 2. the raw numbers ----------------------------------------------------------
pcm16 = np.round(np.clip(x, -1, 1) * 32767).astype(np.int16)  # what a WAV / WebSocket frame carries
peak = np.abs(x).max()
print(f"samples         {len(x):,}  (= {SR:,} per second x {SECONDS} s)")
print(f"as floats       {x[SR:SR + 6]}  ... range {x.min():+.3f} .. {x.max():+.3f}")
print(f"as PCM16 ints   {pcm16[SR:SR + 6]}  (float x 32767)")
print(f"peak level      {20 * np.log10(max(peak, 1e-9)):.1f} dBFS  (0 = loudest possible)")
if peak < 0.01:
    print("                !! almost silent: check System Settings > Privacy > Microphone for your terminal")
for rate in (48_000, 16_000, 8_000):
    print(f"PCM16 @ {rate // 1000:2d} kHz  {rate * 2:,} bytes/s  ({rate * 2 * 60 / 1e6:.2f} MB per minute)")

# where is the energy? (one FFT over the whole clip)
power = np.abs(np.fft.rfft(x)) ** 2
freqs = np.fft.rfftfreq(len(x), 1 / SR)
print("\nenergy by band  ", end="")
for lo, hi in ((0, 1000), (1000, 4000), (4000, 8000), (8000, SR // 2)):
    share = power[(freqs >= lo) & (freqs < hi)].sum() / power.sum() * 100
    print(f"{lo / 1000:g}-{hi / 1000:g} kHz: {share:4.1f}%   ", end="")
print()

# --- 3. resample -----------------------------------------------------------------
x16 = librosa.resample(x, orig_sr=SR, target_sr=16_000)
x8 = librosa.resample(x, orig_sr=SR, target_sr=8_000)
for name, y, rate in (("48k", x, SR), ("16k", x16, 16_000), ("8k", x8, 8_000)):
    sf.write(OUT / f"f8_{name}.wav", y, rate, subtype="PCM_16")
print(f"\nresampled       48k {len(x):,} -> 16k {len(x16):,} -> 8k {len(x8):,} samples; WAVs in {OUT}/")

# --- 4. three views of the same sound --------------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
ax[0].plot(np.arange(len(x)) / SR, x, linewidth=0.5)
ax[0].set(title="1. Waveform (48 kHz): amplitude over time", xlabel="seconds", ylabel="amplitude", xlim=(0, SECONDS))

spec = librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=2048, hop_length=512)), ref=np.max)
img = librosa.display.specshow(spec, sr=SR, hop_length=512, x_axis="time", y_axis="hz", ax=ax[1])
ax[1].set(title="2. Spectrogram (STFT, 48 kHz): which frequencies, when", ylim=(0, 12_000))
fig.colorbar(img, ax=ax[1], format="%+2.0f dB")

# Whisper's front end: 16 kHz, 25 ms window (n_fft 400), 10 ms hop (160), 80 mel bands
logmel = librosa.power_to_db(
    librosa.feature.melspectrogram(y=x16, sr=16_000, n_fft=400, hop_length=160, n_mels=80), ref=np.max
)
img = librosa.display.specshow(logmel, sr=16_000, hop_length=160, x_axis="time", y_axis="mel", ax=ax[2])
ax[2].set(title=f"3. Log-mel, Whisper settings: {logmel.shape[0]} mel bands x {logmel.shape[1]} frames")
fig.colorbar(img, ax=ax[2], format="%+2.0f dB")
fig.savefig(OUT / "f8_voice.png", dpi=110)
print(f"log-mel shape   {logmel.shape}  (mel bands, 10 ms frames) -> plot saved to {OUT}/f8_voice.png")

# --- 5. hear the difference ------------------------------------------------------
for label, y, rate in (("48 kHz original", x, SR), ("16 kHz (what Whisper hears)", x16, 16_000),
                       ("8 kHz (telephone)", x8, 8_000)):
    input(f"\nPress Enter to play {label} ...")
    sd.play(y, rate)
    sd.wait()

print("\nClose the plot window to finish.")
plt.show()
