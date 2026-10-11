# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx", "numpy", "huggingface_hub"]
# ///
"""v0 learn toy: quantise ONE real Qwen weight matrix by hand. Throwaway.

Run: uv run spikes/learn/v0_quant_toy.py   (needs the original Qwen download)
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download

np.set_printoptions(precision=4, suppress=True)

REPO, REV = "Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554"
NAME = "model.layers.0.self_attn.q_proj.weight"


def quantize(w: np.ndarray, bits: int, group: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ruler method, per group: returns (marks, scale, bias) for each group."""
    g = w.reshape(-1, group)  # each row = one group sharing one ruler
    lo, hi = g.min(axis=1, keepdims=True), g.max(axis=1, keepdims=True)
    scale = (hi - lo) / (2**bits - 1)  # gap between marks
    scale[scale == 0] = 1e-12  # a group of identical values: any gap works
    marks = np.round((g - lo) / scale)  # snap to the nearest mark (0 … 2^bits-1)
    return marks, scale, lo


def dequantize(marks: np.ndarray, scale: np.ndarray, bias: np.ndarray, shape: tuple[int, ...]):
    return (marks * scale + bias).reshape(shape)  # start + mark × gap


def rel_error(w: np.ndarray, w2: np.ndarray) -> float:
    """Average error, as % of the average weight size."""
    return float(np.abs(w - w2).mean() / np.abs(w).mean() * 100)


# --- 2b: load one real matrix ---------------------------------------------------
path = Path(snapshot_download(REPO, revision=REV, allow_patterns=["*.json", "*.safetensors"]))
shard = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"][NAME]
w = np.array(mx.load(str(path / shard))[NAME].astype(mx.float32))
print(f"2b  {NAME}")
print(f"    shape {w.shape} = {w.size / 1e6:.1f} M weights, stored as bf16 (16 bits each)")
print(f"    min {w.min():.4f}   max {w.max():.4f}   average |w| {np.abs(w).mean():.4f}")
print(f"    95% of weights lie within ±{np.percentile(np.abs(w), 95):.4f}")

# --- 2c: one group of 64, by hand, at 4 bits ------------------------------------
first = w.reshape(-1)[:64]
marks, scale, bias = quantize(first, bits=4, group=64)
rebuilt = dequantize(marks, scale, bias, first.shape)
print("\n2c  first group of 64 weights, 4-bit (16 marks)")
print(f"    ruler: start {bias.item():.4f}  gap {scale.item():.5f}")
print("    original   mark   rebuilt    error")
for o, m, r in zip(first[:8], marks.reshape(-1)[:8], rebuilt[:8], strict=True):
    print(f"    {o:8.4f}   {int(m):4d}   {r:8.4f}   {o - r:+.4f}")

# --- 2d: whole matrix at different bit widths -----------------------------------
print("\n2d  whole matrix, group 64: average error vs bits")
for bits in (8, 6, 4, 3, 2):
    w2 = dequantize(*quantize(w, bits, 64), w.shape)
    print(f"    {bits}-bit  ({2**bits:3d} marks)  error {rel_error(w, w2):6.2f}%")

# --- 2e: the outlier problem -----------------------------------------------------
groups = w.reshape(-1, 64)
a = np.abs(groups)
spike = a.max(axis=1) / np.median(a, axis=1)  # how far the biggest weight sticks out of its own group
worst, typical = int(spike.argmax()), int(np.argsort(spike)[len(spike) // 2])
print("\n2e  outliers, 4-bit: a typical group vs the group with the sharpest spike")
for label, i in (("typical", typical), ("spike", worst)):
    g = groups[i]
    m, s, b = quantize(g, 4, 64)
    g2 = dequantize(m, s, b, g.shape)
    rest = np.abs(g) < np.abs(g).max()  # the 63 weights other than the biggest
    print(f"    {label:7s}: max/median {spike[i]:5.1f}x  gap {s.item():.4f}  "
          f"error on the other 63 {rel_error(g[rest], g2[rest]):6.2f}%  "
          f"marks they use {len(np.unique(m.reshape(-1)[rest]))}/16")

# --- 2f: group size ---------------------------------------------------------------
print("\n2f  4-bit, different group sizes")
for group in (32, 64, 128, 2560):
    w2 = dequantize(*quantize(w, 4, group), w.shape)
    overhead = 32 / group  # one bf16 scale + one bf16 bias per group, spread over its weights
    label = "whole row" if group == 2560 else f"group {group}"
    print(f"    {label:10s} error {rel_error(w, w2):6.2f}%   storage {4 + overhead:.2f} bits/weight")

# --- 2g: does our hand method match MLX's own quantize? ------------------------
# Not exactly: MLX anchors on the edge furthest from 0 and shrinks the gap so 0 lands on a mark,
# clipping the other edge slightly. Exact zeros, at ~0.2 points more average error.
print("\n2g  our hand-made 4-bit vs MLX's mx.quantize (group 64)")
wq, s, b = mx.quantize(mx.array(w), group_size=64, bits=4)
mlx_rebuilt = np.array(mx.dequantize(wq, s, b, group_size=64, bits=4))
ours = dequantize(*quantize(w, 4, 64), w.shape)
print(f"    MLX error {rel_error(w, mlx_rebuilt):.2f}%   ours {rel_error(w, ours):.2f}%   "
      f"max difference between the two {np.abs(mlx_rebuilt - ours).max():.5f}")
