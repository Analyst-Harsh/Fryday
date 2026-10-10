# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm"]
# ///
"""v0 learn toy: one forward pass gives a next-token guess at EVERY position.

Run: uv run spikes/learn/v0_positions.py      (throwaway learning code)
"""

import mlx.core as mx
from mlx_lm import load

model, tokenizer = load("mlx-community/Qwen3-4B-Instruct-2507-4bit")
messages = [
    {"role": "system", "content": "You are Fryday. Reply in short Hinglish."},
    {"role": "user", "content": "Kal subah 7 baje ka reminder laga do."},
]
ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

# Shapes inside the pass, layer by layer
x = model.model.embed_tokens(mx.array(ids)[None])
print(f"token ids        {(1, len(ids))}")
print(f"after embedding  {tuple(x.shape)}   <- each id became a {x.shape[-1]}-number vector (C)")
print(f"transformer      {len(model.model.layers)} layers, each keeps the shape {tuple(x.shape)}")

logits = model(mx.array(ids)[None])
print(f"after lm_head    {tuple(logits.shape)}   <- C numbers -> one score per vocab token\n")

guess = mx.argmax(logits[0], axis=-1).tolist()  # top-scoring next token at each position
print(f"{'pos':>3}  {'token at this position':24s} {'model guesses next':20s} {'actually next':16s}")
for i, tok in enumerate(ids):
    seen = repr(tokenizer.decode([tok]))
    pred = repr(tokenizer.decode([guess[i]]))
    actual = repr(tokenizer.decode([ids[i + 1]])) if i + 1 < len(ids) else "?? (the reply!)"
    mark = "✓" if i + 1 < len(ids) and guess[i] == ids[i + 1] else " "
    print(f"{i:>3}  {seen:24s} {pred:20s} {actual:16s} {mark}")
