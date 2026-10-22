"""v5 spike, step 1: do Roman replies reach the first speakable chunk sooner than Devanagari?

Same questions, two system prompts (Fryday's Devanagari one, and a Roman copy). Measures
TTFT, first token → first chunk (the real chunker), and exact tokens per word. This is an
upper bound for the Roman option: it leaves out transliteration before TTS (step 2).

Prompts run in blocks, each warmed first, so the server's prefix cache holds the right
system prompt. Block order alternates each round to spread drift.

Needs `make llm` running. Run: uv run python spikes/learn/v5_script_latency.py
"""

import re
import statistics
import time

from fryday.config import Settings
from fryday.llm import LLMClient, Message
from fryday.voice import sentences
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

# The same questions in each script. Option A is Roman end to end (e.g. Oriserve ASR
# outputs Roman), so the Roman prompt gets Roman questions: a Devanagari question pulled
# 19/24 "Roman" replies back into Devanagari (first run, 2026-10-22).
QUESTIONS = {
    "Devanagari": [
        "आज मौसम कैसा है?",
        "मुझे कल सुबह सात बजे याद दिलाना।",
        "एक अच्छी किताब बताओ।",
        "चाय बनाने का आसान तरीका बताओ।",
        "तुम कौन हो?",
        "मेरी मीटिंग कितने बजे है?",
        "थोड़ा मोटिवेशन दो यार।",
        "दिल्ली से मुंबई कितनी दूर है?",
    ],
    "Roman": [
        "aaj mausam kaisa hai?",
        "mujhe kal subah saat baje yaad dilana.",
        "ek achhi kitaab batao.",
        "chai banane ka aasaan tarika batao.",
        "tum kaun ho?",
        "meri meeting kitne baje hai?",
        "thoda motivation do yaar.",
        "Delhi se Mumbai kitni door hai?",
    ],
}
ROUNDS = 3
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")


def roman_prompt(devanagari_prompt: str) -> str:
    old = (
        "Reply in short, friendly Hinglish written only in Devanagari script, including "
        "English words. Write every number, time and amount as Hindi words, never digits "
        "or symbols. "
    )
    new = (
        "Reply in short, friendly Hinglish written only in Roman (English) letters, the "
        "way people text in India, never Devanagari. Write every number, time and amount "
        "as Hindi words in Roman letters, never digits or symbols. "
    )
    if old not in devanagari_prompt:
        raise SystemExit("voice_system_prompt changed: update this spike")
    return devanagari_prompt.replace(old, new)


def run(llm: LLMClient, system: str, question: str) -> dict[str, float | str]:
    history: list[Message] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    t0 = time.perf_counter()
    first_token: list[float] = []  # set on the first delta
    pieces: list[str] = []

    def deltas():
        for d in llm.stream(history):
            if not first_token:
                first_token.append(time.perf_counter())
            pieces.append(d)
            yield d

    chunks = sentences(deltas())
    first_chunk = next(chunks)
    t_chunk = time.perf_counter()
    for _ in chunks:  # drain: full reply for tokens/word and total time
        pass
    return {
        "ttft": first_token[0] - t0,
        "to_chunk": t_chunk - first_token[0],
        "total": time.perf_counter() - t0,
        "chunk": first_chunk,
        "reply": "".join(pieces).strip(),
    }


def main() -> None:
    settings = Settings()
    llm = LLMClient(settings)
    tok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen3-4B-Instruct-2507", "tokenizer.json"))
    prompts = {
        "Devanagari": settings.voice_system_prompt,
        "Roman": roman_prompt(settings.voice_system_prompt),
    }
    rows: dict[str, list[dict[str, float | str]]] = {name: [] for name in prompts}

    for r in range(ROUNDS):
        order = list(prompts) if r % 2 == 0 else list(reversed(prompts))
        for name in order:
            run(llm, prompts[name], QUESTIONS[name][0])  # warm-up: prefix cache, dropped
            for q in QUESTIONS[name]:
                rows[name].append(run(llm, prompts[name], q))
            print(f"round {r + 1}/{ROUNDS}: {name} done")

    print()
    for name, all_rows in rows.items():
        foreign = DEVANAGARI if name == "Roman" else LATIN
        # Only replies in the asked-for script: a slipped reply measures the other script.
        rs = [x for x in all_rows if not foreign.search(str(x["reply"]))]
        chunk_tokens = [len(tok.encode(str(x["chunk"])).ids) for x in rs]
        reply_tokens = [len(tok.encode(str(x["reply"])).ids) for x in rs]
        words = [len(str(x["reply"]).split()) for x in rs]

        def med(key: str, rs: list[dict[str, float | str]] = rs) -> float:
            return statistics.median(float(x[key]) for x in rs) * 1000

        print(
            f"{name:<10} n={len(rs)} (dropped {len(all_rows) - len(rs)} wrong-script)  "
            f"ttft {med('ttft'):4.0f} ms  "
            f"first token → first chunk {med('to_chunk'):4.0f} ms  "
            f"total {med('total'):5.0f} ms  "
            f"chunk tokens {statistics.median(chunk_tokens):.0f}  "
            f"tokens/word {sum(reply_tokens) / sum(words):.2f}"
        )
        for x in all_rows[: len(QUESTIONS[name])]:  # round 1's replies, to eyeball quality
            print(f"    {x['reply']}")
        print()


if __name__ == "__main__":
    main()
