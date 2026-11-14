"""v8 learn 1 toy: export Fryday's Mac embedder to ONNX, look inside, run it in ONNX Runtime.

1. Export multilingual-e5-small (mean pooling + L2 norm inside the graph) with dynamic
   batch and sequence axes.
2. Inspect the file: opset, inputs/outputs, op counts, then the same after ORT's graph
   optimiser (what got fused).
3. Parity: ORT vs PyTorch on the same inputs (max abs diff, cosine).
4. Speed on CPU: PyTorch eager vs ORT, batch 1 and batch 32, same thread count.

`coreml` mode reuses the exported file: ORT's CPU provider vs its CoreML provider
(Apple GPU / Neural Engine), with parity and how much of the graph CoreML took.

Run (deps stay out of the project):
  uv run --with torch --with transformers --with onnx --with onnxscript --with onnxruntime \
    python spikes/learn/v8_onnx_toy.py [coreml]
"""

import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "intfloat/multilingual-e5-small"
OUT = Path(__file__).parent / "out"
ONNX_PATH = OUT / "e5-small.onnx"
OPT_PATH = OUT / "e5-small.opt.onnx"
SENTENCES = [
    "query: आज मौसम कैसा है?",
    "query: mujhe kal subah saat baje yaad dilana",
    "passage: I'm vegetarian and I don't eat eggs.",
    "passage: मेरी मीटिंग शाम पाँच बजे है।",
]


class Embedder(torch.nn.Module):
    """Encoder + mean pooling over real tokens + L2 norm: one embedding per sentence."""

    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, dim=-1)


def op_counts(path: Path) -> Counter[str]:
    return Counter(n.op_type for n in onnx.load(str(path)).graph.node)


def timed(fn, runs: int = 20) -> float:
    """Median ms of `runs` calls after 3 warm-ups."""
    for _ in range(3):
        fn()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Embedder(AutoModel.from_pretrained(MODEL)).eval()

    # 1. Export: trace once with an example batch; mark batch and seq as dynamic.
    ex = tok(SENTENCES[:2], padding=True, return_tensors="pt")
    batch, seq = torch.export.Dim("batch"), torch.export.Dim("seq")
    torch.onnx.export(
        model,
        (ex["input_ids"], ex["attention_mask"]),
        str(ONNX_PATH),
        input_names=["input_ids", "attention_mask"],
        output_names=["sentence_embedding"],
        dynamic_shapes=({0: batch, 1: seq}, {0: batch, 1: seq}),
        dynamo=True,
        external_data=False,  # one file, weights inside
    )

    # 2. Inspect the exported graph.
    m = onnx.load(str(ONNX_PATH))
    print(f"file: {ONNX_PATH.stat().st_size / 1e6:.0f} MB | opset {m.opset_import[0].version}")
    for io in [*m.graph.input, *m.graph.output]:
        dims = [d.dim_param or d.dim_value for d in io.type.tensor_type.shape.dim]
        print(f"  {io.name}: {dims}")
    raw = op_counts(ONNX_PATH)
    print(f"exported graph: {sum(raw.values())} nodes, top ops {raw.most_common(6)}")

    # ORT's optimiser, saved to a file so we can see what it fused.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = str(OPT_PATH)
    # Same thread count as PyTorch: ORT's default (all 10 cores, incl. efficiency cores)
    # made batch 32 look 0.76x in the first run.
    so.intra_op_num_threads = torch.get_num_threads()
    sess = ort.InferenceSession(str(ONNX_PATH), so, providers=["CPUExecutionProvider"])
    opt = op_counts(OPT_PATH)
    print(f"after ORT opt:  {sum(opt.values())} nodes, top ops {opt.most_common(6)}")

    # 3. Parity on all sentences (a batch size the trace never saw).
    enc = tok(SENTENCES, padding=True, return_tensors="pt")
    feeds = {k: enc[k].numpy() for k in ("input_ids", "attention_mask")}
    with torch.inference_mode():
        pt = model(enc["input_ids"], enc["attention_mask"]).numpy()
    ox = sess.run(["sentence_embedding"], feeds)[0]
    cos = (pt * ox).sum(-1)  # both are unit vectors
    print(f"parity: max |diff| {np.abs(pt - ox).max():.2e}, min cosine {cos.min():.8f}")

    # 4. Speed, same thread count for both.
    print(f"threads: torch {torch.get_num_threads()}, ORT {so.intra_op_num_threads}")
    for n in (1, 32):
        enc = tok((SENTENCES * 8)[:n], padding=True, return_tensors="pt")
        feeds = {k: enc[k].numpy() for k in ("input_ids", "attention_mask")}
        with torch.inference_mode():
            t_pt = timed(lambda e=enc: model(e["input_ids"], e["attention_mask"]))
        t_ox = timed(lambda f=feeds: sess.run(["sentence_embedding"], f))
        print(f"batch {n:>2}: PyTorch {t_pt:6.1f} ms | ORT {t_ox:6.1f} ms | {t_pt / t_ox:.2f}x")


def coreml() -> None:
    """Same .onnx, two execution providers: CPU vs CoreML (falls back to CPU per node)."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    so = ort.SessionOptions()
    so.intra_op_num_threads = torch.get_num_threads()
    cpu = ort.InferenceSession(str(ONNX_PATH), so, providers=["CPUExecutionProvider"])
    so_cml = ort.SessionOptions()
    so_cml.intra_op_num_threads = torch.get_num_threads()
    so_cml.log_severity_level = 1  # INFO: CoreML logs how many nodes it supports
    t = time.perf_counter()
    cml = ort.InferenceSession(
        str(ONNX_PATH),
        so_cml,
        providers=[
            ("CoreMLExecutionProvider", {"ModelFormat": "NeuralNetwork", "MLComputeUnits": "ALL"}),
            "CPUExecutionProvider",
        ],
    )
    print(f"CoreML session load: {time.perf_counter() - t:.1f} s")

    for n in (1, 32):
        enc = tok((SENTENCES * 8)[:n], padding=True, return_tensors="np")
        feeds = {k: enc[k] for k in ("input_ids", "attention_mask")}
        a = cpu.run(["sentence_embedding"], feeds)[0]
        b = cml.run(["sentence_embedding"], feeds)[0]
        t_cpu = timed(lambda f=feeds: cpu.run(["sentence_embedding"], f))
        t_cml = timed(lambda f=feeds: cml.run(["sentence_embedding"], f))
        print(
            f"batch {n:>2}: CPU {t_cpu:6.1f} ms | CoreML {t_cml:6.1f} ms | {t_cpu / t_cml:.2f}x"
            f" | max |diff| {np.abs(a - b).max():.1e}, min cos {(a * b).sum(-1).min():.6f}"
        )


if __name__ == "__main__":
    coreml() if sys.argv[1:] == ["coreml"] else main()
