# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "torchaudio", "librosa", "mlx-whisper==0.4.3", "jiwer==4.0.0", "huggingface_hub"]
# ///
"""v3 learn 2 toy: score the learn-1 Kokoro WAVs two ways and compare with the owner's ears. Throwaway.

1. Naturalness: UTMOS22 predicted MOS (no text, no reference: "does it sound human?").
2. Intelligibility: ASR round trip. Whisper transcribes the TTS audio; CER vs the intended spoken text,
   plus a targeted check that the words that matter (time, money) came back.

Needs the WAVs from spikes/learn/v3_tts_toy.py. Run from the repo root: uv run spikes/learn/v3_tts_eval.py
"""

import re
import time
from pathlib import Path

import jiwer
import librosa
import mlx_whisper
import torch
from huggingface_hub import snapshot_download

AUDIO = Path("data/audio")
# torch.hub runs code from GitHub, so pin the commit (tag v1.2.0), not a branch
UTMOS = "tarepan/SpeechMOS:ed25eacbfa42b99156c36ebec67a733b5dbb9b79"
WHISPER, WHISPER_REV = "mlx-community/whisper-large-v3-turbo", "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"

# (file, whisper language, reference = what we MEANT to be spoken, words that must survive, owner's ears); "a|b" = either form
# hi_roman's reference is Devanagari: same meaning, and Whisper writes Hindi in Devanagari.
CASES = [
    ("en", "en", "hi i'm fryday your gym reminder is set for seven a m tomorrow", ["seven|7"], "—"),
    ("hi_deva", "hi", "कल सुबह सात बजे जिम है मैं आपको याद दिला दूँगी", ["सात"], "spot on"),
    ("hi_roman", "hi", "कल सुबह सात बजे जिम है मैं आपको याद दिला दूँगी", ["सात"], "weak"),
    ("hi_deva_male", "hi", "कल सुबह सात बजे जिम है मैं आपको याद दिला दूँगा", ["सात"], "—"),
    ("hi_digits", "hi", "कल सुबह सात बजे जिम है और दो सौ पचास रुपये का ऑर्डर है", ["सात", "रुपये"], "best-sounding"),
]


def norm(text: str) -> str:
    """Same cleanup on both sides, so CER measures speech, not formatting."""
    text = text.lower().replace("ँ", "ं").replace("रुपए", "रुपये")  # common Hindi spelling variants
    # punctuation, incl. । and the "." in "a.m.". Not [^\w]: \w drops Hindi vowel signs (सुबह -> स बह)
    text = re.sub(r"[.,!?;:।\"“”‘’()-]", " ", text)
    return " ".join(text.split())


t0 = time.perf_counter()
mos = torch.hub.load(UTMOS, "utmos22_strong", trust_repo=True)
whisper = snapshot_download(WHISPER, revision=WHISPER_REV)
print(f"loaded UTMOS22 + {WHISPER}@{WHISPER_REV[:7]} in {time.perf_counter() - t0:.1f} s\n")

rows = []
for name, lang, ref, must, ears in CASES:
    wav = AUDIO / f"v3_{name}.wav"
    x, sr = librosa.load(wav, sr=None, mono=True)  # Kokoro writes 24 kHz; UTMOS resamples internally
    score = mos(torch.from_numpy(x).unsqueeze(0), sr).item()
    # an array (not a path) at 16 kHz: a path makes mlx-whisper shell out to ffmpeg
    heard = mlx_whisper.transcribe(librosa.resample(x, orig_sr=sr, target_sr=16_000),
                                   path_or_hf_repo=whisper, language=lang)["text"]
    cer = jiwer.cer(norm(ref), norm(heard))
    missing = [w for w in must if not set(norm(w).split("|")) & set(norm(heard).split())]  # "a|b" = either
    rows.append((name, score, cer, missing, ears))
    print(f"[{name}]\n  meant  {norm(ref)}\n  heard  {norm(heard)}\n"
          f"  UTMOS {score:.2f} | CER {cer:.0%} | missing {missing or 'none'}\n")

print(f"{'file':14s} {'UTMOS':>6s} {'CER':>5s}  {'key words missing':20s} owner's ears")
for name, score, cer, missing, ears in rows:
    print(f"{name:14s} {score:6.2f} {cer:5.0%}  {', '.join(missing) or 'none':20s} {ears}")
print("\nUTMOS: higher = more natural (1-5, trained on English). CER: lower = more intelligible "
      "(includes Whisper's own errors).")
