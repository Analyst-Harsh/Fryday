# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///
"""F session 2 toy: one attention head, by hand. Throwaway learning code.

Run: uv run spikes/learn/f2_attention.py
"""

import numpy as np

np.set_printoptions(precision=2, suppress=True)

# Hand-made 3-d embeddings. Features: [water, money, plain-word]
EMB = {
    "the": [0.0, 0.0, 1.0],
    "my": [0.0, 0.0, 1.0],
    "river": [1.0, 0.0, 0.2],
    "money": [0.0, 1.0, 0.2],
    "bank": [0.5, 0.5, 0.3],  # ambiguous: half water, half money
}


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def attend(sentence):
    X = np.array([EMB[w] for w in sentence])  # (tokens, dim)
    # Real models learn W_q, W_k, W_v. Identity keeps the maths visible.
    Q, K, V = X, X, X
    q = Q[-1]  # the last token ("bank") asks: who matters to me?
    scores = K @ q / np.sqrt(X.shape[1])  # dot product = similarity
    weights = softmax(scores)  # turn scores into a 0-1 mix
    new_bank = weights @ V  # blend the values by those weights
    print(f"\n{' '.join(sentence)}")
    for w, a in zip(sentence, weights):
        print(f"  attention bank->{w:6s} {a:.2f}")
    print(f"  bank before: {X[-1]}   after: {new_bank}   [water, money, plain]")


attend(["the", "river", "bank"])
attend(["my", "money", "bank"])

# Last step of a transformer: logits -> softmax -> probabilities over the vocab.
print("\nnext-token probabilities from logits [2.0, 1.0, 0.1]:")
for t in (0.5, 1.0, 2.0):
    print(f"  temperature {t}: {softmax(np.array([2.0, 1.0, 0.1]) / t)}")
