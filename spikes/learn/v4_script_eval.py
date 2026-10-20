# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "mlx-whisper==0.4.3", "transformers>=4.45,<5", "torch", "jiwer==4.0.0", "huggingface_hub",
#   "pyarrow", "soundfile", "indic-transliteration==2.3.69",
# ]
# ///
"""v4 build (S0-8): Whisper → Devanagari vs Oriserve → Roman Hinglish, on MUCS Hinglish clips.

The two models write different scripts, and MUCS references are mixed (Hindi in Devanagari,
English words in Latin). Plain WER would score script, not accuracy. So every text is reduced
to a "sound skeleton": romanised (IAST), accents stripped, vowels and h dropped, repeats
collapsed. "tutorial", "टुटोरियल" and "tutoriyal" all become "ttrl". It's crude, but it
treats both models the same. Score = CER on skeletons.

Run from the repo root: uv run spikes/learn/v4_script_eval.py   (downloads ~2 GB the first time)
"""

import io
import random
import re
import statistics
import time
import unicodedata

import jiwer
import mlx_whisper
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from indic_transliteration import sanscript
from transformers import pipeline

N = 100
DATA, DATA_REV = "dianavdavidson/MUCS-Hinglish", "3c66ee127a7b0e0a746d4df1002858567135edc1"
WHISPER, WHISPER_REV = "mlx-community/whisper-large-v3-turbo", "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
ORISERVE, ORISERVE_REV = "Oriserve/Whisper-Hindi2Hinglish-Apex", "f3214eed20b4e4d4144e739982d911f87b9cb223"
# Same loop guard mlx_whisper.transcribe() applies by default: if a decode looks repetitive
# (compression ratio) or unsure (avg logprob), retry at a higher temperature. HF measures the
# ratio on tokens, so its recommended threshold is 1.35, not 2.4.
ORISERVE_GEN = {
    "task": "transcribe", "language": "en", "condition_on_prev_tokens": False,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "compression_ratio_threshold": 1.35, "logprob_threshold": -1.0, "no_speech_threshold": 0.6,
}
DEVANAGARI = re.compile(r"[ऀ-ॿ]+")


def skeleton(text: str) -> str:
    text = DEVANAGARI.sub(lambda m: sanscript.transliterate(m[0], sanscript.DEVANAGARI, sanscript.IAST), text)
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z ]", " ", text)
    text = re.sub(r"[aeiouyh]", "", text)  # vowels and h: where the two scripts disagree most
    text = text.translate(str.maketrans("wzfq", "vjpk"))  # sounds Hinglish spells two ways
    text = re.sub(r"(.)\1+", r"\1", text)
    return " ".join(text.split())


def load_clips() -> list[dict]:
    path = hf_hub_download(DATA, "data/test-00000-of-00002.parquet", repo_type="dataset", revision=DATA_REV)
    rows = pq.read_table(path, columns=["audio", "transcript"]).to_pylist()
    random.Random(0).shuffle(rows)
    clips = []
    for r in rows[:N]:
        audio, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        assert sr == 16_000
        clips.append({"audio": audio, "ref": r["transcript"]})
    return clips


def run(name: str, transcribe, clips: list[dict]) -> list[str]:
    transcribe(clips[0]["audio"])  # warm-up
    outs, times = [], []
    for c in clips:
        t0 = time.perf_counter()
        outs.append(transcribe(c["audio"]).strip())
        times.append(time.perf_counter() - t0)
    print(f"{name}: median {statistics.median(times):.2f} s/clip")
    return outs


def main() -> None:
    clips = load_clips()
    secs = [len(c["audio"]) / 16_000 for c in clips]
    print(f"{N} clips, {min(secs):.1f}-{max(secs):.1f} s, median {statistics.median(secs):.1f} s")

    wpath = snapshot_download(WHISPER, revision=WHISPER_REV)
    whisper = run("Whisper (hi)", lambda a: mlx_whisper.transcribe(
        a, path_or_hf_repo=wpath, language="hi", condition_on_previous_text=False)["text"], clips)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = pipeline("automatic-speech-recognition", model=ORISERVE, revision=ORISERVE_REV,
                    dtype=torch.float16, device=device)  # fp32 gave identical scores (EXPERIMENTS)
    oriserve = run(f"Oriserve ({device})", lambda a: pipe(
        np.asarray(a), return_timestamps=True, generate_kwargs=ORISERVE_GEN)["text"], clips)

    refs = [skeleton(c["ref"]) for c in clips]
    for name, outs in [("Whisper → Devanagari", whisper), ("Oriserve → Roman", oriserve)]:
        hyps = [skeleton(o) for o in outs]
        per = [jiwer.cer(r, h) if r else 0.0 for r, h in zip(refs, hyps, strict=True)]
        wins = sum(p < q for p, q in zip(per, [jiwer.cer(r, skeleton(x)) for r, x in
                   zip(refs, oriserve if outs is whisper else whisper, strict=True)], strict=True))
        print(f"{name}: skeleton CER {jiwer.cer(refs, hyps):.1%} (pooled), "
              f"median {statistics.median(per):.1%}, better on {wins}/{N} clips")

    print("\nExamples (ref / whisper / oriserve):")
    for c, w, o in list(zip(clips, whisper, oriserve, strict=True))[:5]:
        print(f"  R: {c['ref']}\n  W: {w}\n  O: {o}\n")


if __name__ == "__main__":
    main()
