# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm"]
# ///
"""v0 learn toy: look at Qwen's raw next-token scores (logits) and how sampling reshapes them.

Run: uv run spikes/learn/v0_logits.py      (throwaway learning code)
"""

import mlx.core as mx
from mlx_lm import load

MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
model, tokenizer = load(MODEL)

messages = [
    {"role": "system", "content": "You are Fryday. Reply in short Hinglish."},
    {"role": "user", "content": "Kal subah 7 baje ka reminder laga do."},
]
ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

# One forward pass over the whole prompt (= prefill). Output: one score per vocab entry
# for EVERY position; we only need the last position -> "what comes next?"
logits = model(mx.array(ids)[None])  # shape (B=1, T, vocab_size)
print(f"prompt tokens T={len(ids)}   logits shape={tuple(logits.shape)}")
last = logits[0, -1].astype(mx.float32)  # (vocab_size,)


def top(probs, k=8):
    order = mx.argsort(-probs)[:k].tolist()
    return [(repr(tokenizer.decode([i])), round(probs[i].item(), 3)) for i in order]


print("\nraw logits (top 8 by score):")
order = mx.argsort(-last)[:8].tolist()
print([(repr(tokenizer.decode([i])), round(last[i].item(), 2)) for i in order])

for t in (0.3, 0.7, 1.0, 1.5):
    probs = mx.softmax(last / t)
    print(f"\ntemperature {t}: top-8 probabilities")
    print(top(probs))
    print(f"  -> tokens needed to cover 90% of probability (what top-p=0.9 keeps): ", end="")
    sorted_p = mx.sort(probs)[::-1]
    print(int((mx.cumsum(sorted_p) < 0.9).sum().item()) + 1)
