# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm"]
# ///
"""v0 learn: bf16 vs 8/6/4-bit Qwen3-4B on the Mac: memory, TTFT, decode speed, agreement with bf16.

Needs the original Qwen download and data/models/qwen3-4b-{8,6,4}bit (build 4 step 3).
Run: uv run spikes/learn/v0_quant.py   (throwaway learning code)
"""

import gc
import time
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import load, stream_generate

BANDWIDTH = 153e9  # M5 unified-memory bandwidth, bytes/s
MAX_TOKENS = 64
ORIGINAL = snapshot_download("Qwen/Qwen3-4B-Instruct-2507", revision="cdbee75f17c01a7cc42f958dc650907174af0554",
                             allow_patterns=["*.json", "*.safetensors", "*.txt"])
VARIANTS = {"bf16": ORIGINAL, **{f"{b}-bit": f"data/models/qwen3-4b-{b}bit" for b in (8, 6, 4)}}
SYSTEM = "You are Fryday, a helpful assistant. Reply in short, friendly Hinglish."
PROMPTS = [
    "Kal subah 7 baje mujhe gym ke liye yaad dilana.",
    "Aaj Bangalore ka mausam kaisa rahega?",
    "Mujhe ek simple dal recipe batao, jaldi wali.",
    "Meri mummy ka birthday next week hai, gift ideas do.",
    "Python mein list aur tuple mein kya difference hai?",
    "Mera mood thoda off hai, kuch motivate karo.",
    "Shopping list mein doodh, bread aur ande add kar do.",
    "Ek joke sunao, but clean wala.",
]


def prompt_ids(tokenizer, user: str) -> list[int]:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def run(model, tokenizer, user: str, max_tokens: int = MAX_TOKENS) -> tuple[list[int], float, float]:
    """Greedy reply (mlx_lm's default sampler is argmax): (token ids, TTFT seconds, decode tok/s)."""
    prompt = prompt_ids(tokenizer, user)
    start, ttft, tokens = time.perf_counter(), 0.0, []
    for r in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        if not tokens:
            ttft = time.perf_counter() - start
        tokens.append(r.token)
    return tokens, ttft, r.generation_tps


def forced(model, tokenizer, user: str, reply: list[int]) -> int:
    """Teacher forcing: feed bf16's reply in, count positions where this model's top token is bf16's next token."""
    ids = prompt_ids(tokenizer, user)
    logits = model(mx.array([ids + reply[:-1]]))[0, len(ids) - 1 :]  # position k predicts reply[k]
    return int((mx.argmax(logits, axis=-1) == mx.array(reply)).sum().item())


def matched(a: list[int], b: list[int]) -> int:
    """How many tokens agree before the first difference."""
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


reference: list[list[int]] = []
rows = []
for name, path in VARIANTS.items():
    weight_bytes = sum(p.stat().st_size for p in Path(path).glob("*.safetensors"))
    print(f"\n{'=' * 20} {name}  ({weight_bytes / 1e9:.2f} GB of weights) {'=' * 20}", flush=True)
    mx.reset_peak_memory()
    model, tokenizer = load(path)
    run(model, tokenizer, "Hi", max_tokens=8)  # warm-up: first call compiles kernels
    replies, ttfts, speeds = [], [], []
    for i, user in enumerate(PROMPTS):
        tokens, ttft, tps = run(model, tokenizer, user)
        replies.append(tokens)
        ttfts.append(ttft)
        speeds.append(tps)
        vs_bf16 = f"matches bf16 for {matched(tokens, reference[i])}/{len(tokens)} tokens" if reference else "reference"
        text = tokenizer.decode(tokens).replace("<|im_end|>", "").strip().replace("\n", "\n      ")
        print(f"\n  [{i + 1}] {user}\n      {text}\n      ({ttft * 1000:.0f} ms TTFT, {tps:.1f} tok/s, {vs_bf16})",
              flush=True)
    peak = mx.get_peak_memory() / 1e9  # before the teacher-forcing pass, which holds full logits
    reference = reference or replies  # bf16 runs first and is the reference
    agree = sum(forced(model, tokenizer, u, ref) for u, ref in zip(PROMPTS, reference, strict=True))
    agree_pct = agree / sum(len(ref) for ref in reference) * 100
    same = [matched(r, ref) for r, ref in zip(replies, reference, strict=True)]
    identical = sum(r == ref for r, ref in zip(replies, reference, strict=True))
    rows.append((name, weight_bytes / 1e9, peak, sum(ttfts) / len(ttfts) * 1000,
                 sum(speeds) / len(speeds), BANDWIDTH / weight_bytes, identical, sum(same) / len(same), agree_pct))
    del model, tokenizer
    gc.collect()
    mx.clear_cache()

print(f"\n{'variant':7s} {'weights':>8s} {'peak mem':>9s} {'TTFT':>7s} {'decode':>9s} {'ceiling':>8s} "
      f"{'% of ceil':>9s} {'identical':>9s} {'avg match':>9s} {'forced':>7s}")
for name, gb, peak, ttft, tps, ceil, identical, avg, agree_pct in rows:
    print(f"{name:7s} {gb:6.2f}GB {peak:7.2f}GB {ttft:5.0f}ms {tps:5.1f}t/s {ceil:5.1f}t/s "
          f"{tps / ceil * 100:8.0f}% {identical:5d}/{len(PROMPTS)} {avg:6.1f}/{MAX_TOKENS} {agree_pct:6.1f}%")
