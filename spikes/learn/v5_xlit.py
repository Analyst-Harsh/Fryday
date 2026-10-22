"""v5 spike, step 2: Roman replies → IndicXlit → Devanagari → Kokoro. Is it fast and clean?

Step 1 (v5_script_latency.py) found Roman replies reach the first chunk 251 ms sooner.
This measures the cost Roman adds back: transliterating each chunk before TTS (and the
input side, Whisper's Devanagari → Roman). It also writes a blind A/B listening test:
for each question, the Roman pipeline vs today's Devanagari pipeline, randomly labelled.

IndicXlit (AI4Bharat, MIT) via a community CTranslate2 port: no fairseq. Word level,
so punctuation and digits pass through; normalise() turns digits into words after.

Needs `make llm` running. Run (no new project deps):
    uv run --with ctranslate2 python ../spikes/learn/v5_xlit.py <out_dir>
"""

import random
import re
import statistics
import sys
import time
import wave
from pathlib import Path

import ctranslate2
from fryday.config import Settings
from fryday.llm import LLMClient, Message
from fryday.normalise import normalise
from fryday.tts import TTSClient
from fryday.voice import sentences
from huggingface_hub import snapshot_download
from v5_script_latency import QUESTIONS, roman_prompt

REPO, REVISION = "Singla0009/all-indic-transliteration", "23279e441f195f6a45296fcee84ac89b541254ff"
ROMAN_WORD = re.compile(r"[A-Za-z]+")
DEVANAGARI_WORD = re.compile(r"[ऀ-ॿ]+")


class Xlit:
    """One direction of IndicXlit: every word matching `word` is replaced, in one batch."""

    def __init__(self, folder: str, word: re.Pattern[str]) -> None:
        path = snapshot_download(REPO, revision=REVISION, allow_patterns=f"{folder}/*")
        self._model = ctranslate2.Translator(f"{path}/{folder}", device="cpu")
        self._word = word

    def __call__(self, text: str) -> str:
        words = list(dict.fromkeys(m.group().lower() for m in self._word.finditer(text)))
        if not words:
            return text
        batch = [["__hi__", *w] for w in words]
        results = self._model.translate_batch(batch, beam_size=4)
        out = {w: "".join(r.hypotheses[0]) for w, r in zip(words, results, strict=True)}
        return self._word.sub(lambda m: out[m.group().lower()], text)


def reply_chunks(llm: LLMClient, system: str, question: str) -> list[str]:
    history: list[Message] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    return list(sentences(llm.stream(history)))


def write_wav(path: Path, tts: TTSClient, chunks: list[str]) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(tts.sample_rate)
        for chunk in chunks:
            w.writeframes(b"".join(tts.speak(normalise(chunk))))


def timed(fn, text: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    out = fn(text)
    return out, (time.perf_counter() - t0) * 1000


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    llm, tts = LLMClient(settings), TTSClient(settings)
    to_dev = Xlit("indicxlit_ct2_fp32", ROMAN_WORD)
    to_roman = Xlit("indicxlit_indic_en_ct2_fp32", DEVANAGARI_WORD)
    to_dev("namaste dost"), to_roman("नमस्ते दोस्त")  # warm-up
    list(tts.speak("नमस्ते।"))
    prompts = {
        "Devanagari": settings.voice_system_prompt,
        "Roman": roman_prompt(settings.voice_system_prompt),
    }
    reply_chunks(llm, prompts["Roman"], QUESTIONS["Roman"][0])  # warm the prefix cache

    first_ms, chunk_ms, word_ms, roman_replies = [], [], [], []
    for q in QUESTIONS["Roman"]:
        chunks = reply_chunks(llm, prompts["Roman"], q)
        converted = []
        for i, chunk in enumerate(chunks):
            dev, ms = timed(to_dev, chunk)
            converted.append(dev)
            chunk_ms.append(ms)
            word_ms.append(ms / max(1, len(chunk.split())))
            if i == 0:
                first_ms.append(ms)
        roman_replies.append((q, chunks, converted))

    reply_chunks(llm, prompts["Devanagari"], QUESTIONS["Devanagari"][0])  # prefix cache
    native = [reply_chunks(llm, prompts["Devanagari"], q) for q in QUESTIONS["Devanagari"]]

    input_ms = []
    print("Input side: Whisper-style Devanagari question → Roman")
    for q in QUESTIONS["Devanagari"]:
        roman, ms = timed(to_roman, q)
        input_ms.append(ms)
        print(f"    {ms:4.1f} ms  {q}  →  {roman}")

    print("\nRoman reply → Devanagari (what Kokoro reads)")
    for q, chunks, converted in roman_replies:
        print(f"  Q: {q}")
        for r, d in zip(chunks, converted, strict=True):
            print(f"    {r}\n    {d}")

    # Blind A/B: which file is which is only in key.txt.
    rng = random.Random(0)  # noqa: S311  # shuffling a listening test, not crypto
    key = []
    for n, ((_, _, converted), dev_chunks) in enumerate(zip(roman_replies, native, strict=True), 1):
        pair = [("roman+xlit", converted), ("devanagari", dev_chunks)]
        rng.shuffle(pair)
        for label, (name, chunks) in zip("AB", pair, strict=True):
            write_wav(out_dir / f"q{n}_{label}.wav", tts, chunks)
            key.append(f"q{n}_{label}: {name}")
    (out_dir / "key.txt").write_text("\n".join(key) + "\n")

    def stats(ms: list[float]) -> str:
        return f"median {statistics.median(ms):.1f} ms, max {max(ms):.1f} ms (n={len(ms)})"

    print(
        f"\nxlit first chunk: {stats(first_ms)}"
        f"\nxlit any chunk:   {stats(chunk_ms)}"
        f"  ≈ {statistics.median(word_ms):.2f} ms/word"
        f"\ninput → Roman:    {stats(input_ms)}"
        f"\nWAVs: {out_dir} (q1_A … q8_B; answers in key.txt)"
    )


if __name__ == "__main__":
    main()
