# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-audio", "misaki[en]", "huggingface_hub", "numpy", "soundfile", "sounddevice"]
# ///
"""v3 learn 1 toy: make the Mac speak with Kokoro-82M. Throwaway learning code.

For each sentence: show the phonemes (G2P), generate speech, measure time-to-first-audio and RTF,
save a WAV to data/audio/ (gitignored), and play it. Covers English, Hindi in Devanagari vs Roman,
and un-normalised digits / rupee sign.

Run from the repo root: uv run spikes/learn/v3_tts_toy.py
"""

import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import sounddevice as sd
import soundfile as sf
from huggingface_hub import snapshot_download
from mlx_audio.tts.utils import load_model

MODEL, REV = "mlx-community/Kokoro-82M-bf16", "a71e4d38b236d968966a2002c4c895dbd12b1c3c"
VOICES = ["af_heart", "hf_alpha", "hm_omega"]  # US English female, Hindi female, Hindi male
OUT = Path("data/audio")
OUT.mkdir(parents=True, exist_ok=True)
SENTENCES = [  # (file name, Kokoro lang code, voice, text)
    ("en", "a", "af_heart", "Hi, I'm Fryday. Your gym reminder is set for seven a.m. tomorrow."),
    ("hi_deva", "h", "hf_alpha", "कल सुबह सात बजे जिम है। मैं आपको याद दिला दूँगी।"),
    ("hi_roman", "h", "hf_alpha", "Kal subah saat baje gym hai. Main aapko yaad dila doongi."),
    ("hi_deva_male", "h", "hm_omega", "कल सुबह सात बजे जिम है। मैं आपको याद दिला दूँगा।"),
    ("hi_digits", "h", "hf_alpha", "कल सुबह 7 बजे जिम है, और ₹250 का ऑर्डर है।"),
]

# Pin model AND voices to one revision: by name, mlx-audio would fetch voices unpinned from another repo.
path = Path(snapshot_download(MODEL, revision=REV, allow_patterns=["*.json", "kokoro-v1_0.safetensors",
                                                                    *[f"voices/{v}.safetensors" for v in VOICES]]))
mx.reset_peak_memory()
t0 = time.perf_counter()
model = load_model(path)
print(f"loaded {MODEL}@{REV[:7]} in {time.perf_counter() - t0:.1f} s, sample rate {model.sample_rate} Hz\n")


def speak(text: str, lang: str, voice: str) -> tuple[np.ndarray, float, float]:
    """Generate one utterance: (audio, seconds to first audio chunk, total seconds)."""
    start, first, chunks = time.perf_counter(), 0.0, []
    for result in model.generate(text, voice=str(path / "voices" / f"{voice}.safetensors"), lang_code=lang):
        if not chunks:
            first = time.perf_counter() - start
        chunks.append(np.array(result.audio))
    return np.concatenate(chunks), first, time.perf_counter() - start


speak("Warm up.", "a", "af_heart")  # first call compiles GPU kernels; don't time it

rows = []
for name, lang, voice, text in SENTENCES:
    phonemes = model._get_pipeline(lang).g2p(text)[0]  # private API: show what the network actually reads
    audio, first, total = speak(text, lang, voice)
    seconds = len(audio) / model.sample_rate
    sf.write(OUT / f"v3_{name}.wav", audio, model.sample_rate, subtype="PCM_16")
    rows.append((name, voice, seconds, first, total))
    print(f"[{name}] {voice}\n  text      {text}\n  phonemes  {phonemes}\n"
          f"  audio {seconds:.2f} s | first audio {first * 1000:.0f} ms | total {total * 1000:.0f} ms | "
          f"RTF {total / seconds:.3f}")
    input("  Enter to play ...")
    sd.play(audio, model.sample_rate)
    sd.wait()
    print()

print(f"{'sentence':14s} {'voice':9s} {'audio':>7s} {'1st audio':>10s} {'RTF':>6s}")
for name, voice, seconds, first, total in rows:
    print(f"{name:14s} {voice:9s} {seconds:6.2f}s {first * 1000:8.0f}ms {total / seconds:6.3f}")
print(f"\npeak memory {mx.get_peak_memory() / 1e9:.2f} GB   WAVs in {OUT}/")
