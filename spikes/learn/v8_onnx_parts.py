"""v8 learn 1: the parts of an ONNX file, on a model small enough to read in full.

  step 1  export Tiny (Linear 3→2 + ReLU) with a FIXED shape; print every part of the file
  step 2  dynamic axes: run the fixed file with another batch size, then re-export with a
          dynamic batch axis and run it again
  step 3  opset: export a LayerNorm at an old and a new opset and compare the nodes

Run one step at a time:
  uv run --with torch --with onnx --with onnxscript --with onnxruntime \
    python spikes/learn/v8_onnx_parts.py 1
"""

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper

OUT = Path(__file__).parent / "out"


class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)  # W: 2×3, b: 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


def shape(value_info: onnx.ValueInfoProto) -> list[str | int]:
    """[dims] of a graph input/output: a number (fixed) or a name (dynamic)."""
    return [d.dim_param or d.dim_value for d in value_info.type.tensor_type.shape.dim]


def show(path: Path) -> None:
    """Print every part of an ONNX file."""
    m = onnx.load(str(path))
    g = m.graph
    print(f"\n== {path.name} ({path.stat().st_size} bytes) ==")
    print("opset:      ", {o.domain or "ai.onnx": o.version for o in m.opset_import})
    print("inputs:     ", [(i.name, shape(i)) for i in g.input])
    print("outputs:    ", [(o.name, shape(o)) for o in g.output])
    print("initializers (weights stored in the file):")
    for w in g.initializer:
        arr = numpy_helper.to_array(w)
        print(f"   {w.name}  shape {list(arr.shape)}  values {np.round(arr, 3).tolist()}")
    print(f"nodes ({len(g.node)}), in order:")
    for n in g.node:
        print(f"   {n.op_type:<20} in {list(n.input)} → out {list(n.output)}")


def step1() -> None:
    torch.manual_seed(0)
    model = Tiny().eval()
    x = torch.randn(2, 3)  # example input: batch 2
    path = OUT / "tiny_fixed.onnx"
    torch.onnx.export(model, (x,), str(path), input_names=["x"], output_names=["y"], dynamo=True)
    show(path)

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print("\nPyTorch:", model(x).detach().numpy().round(4).tolist())
    print("ORT:    ", sess.run(["y"], {"x": x.numpy()})[0].round(4).tolist())


def step2() -> None:
    torch.manual_seed(0)
    model = Tiny().eval()
    x4 = torch.randn(4, 3)

    # a) The fixed file from step 1 was traced with batch 2. Send it batch 4.
    fixed = ort.InferenceSession(str(OUT / "tiny_fixed.onnx"), providers=["CPUExecutionProvider"])
    print("fixed file expects:", [(i.name, i.shape) for i in fixed.get_inputs()])
    try:
        fixed.run(["y"], {"x": x4.numpy()})
        print("batch 4 → worked")
    except Exception as e:  # ORT raises its own error types
        print(f"batch 4 → {type(e).__name__}: {str(e)[:160]}")

    # b) Re-export with dimension 0 of x marked dynamic.
    path = OUT / "tiny_dynamic.onnx"
    torch.onnx.export(
        model,
        (torch.randn(2, 3),),
        str(path),
        input_names=["x"],
        output_names=["y"],
        dynamic_shapes=({0: torch.export.Dim("batch")},),
        dynamo=True,
    )
    show(path)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for n in (1, 4, 32):
        y = sess.run(["y"], {"x": torch.randn(n, 3).numpy()})[0]
        print(f"batch {n:>2} → output shape {list(y.shape)}")


def step3() -> None:
    # LayerNormalization joined the ONNX op catalogue in opset 17.
    model = torch.nn.LayerNorm(4).eval()
    x = torch.randn(2, 4)
    # The new (dynamo) exporter only targets opset 18+, so opset 16 uses the legacy one.
    for opset, dynamo in ((16, False), (20, True)):
        path = OUT / f"layernorm_opset{opset}.onnx"
        torch.onnx.export(
            model,
            (x,),
            str(path),
            input_names=["x"],
            output_names=["y"],
            opset_version=opset,
            dynamo=dynamo,
        )
        show(path)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        diff = np.abs(sess.run(["y"], {"x": x.numpy()})[0] - model(x).detach().numpy()).max()
        print(f"max |diff| vs PyTorch: {diff:.1e}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    {"1": step1, "2": step2, "3": step3}[sys.argv[1] if len(sys.argv) > 1 else "1"]()
