"""v8 S0-1 check: is Triton serving e5 correctly? The owner's config.pbtxt must pass this.

1. Server ready, model `e5` ready (KServe v2 health endpoints).
2. One request over HTTP (KServe v2 JSON) with 4 sentences of different lengths.
3. Parity: Triton's embeddings vs ONNX Runtime run locally on the same file.

Needs Triton running with infra/triton/model_repository mounted, HTTP on :8010.
  uv run --with transformers --with onnxruntime python spikes/learn/v8_triton_check.py
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

URL = "http://localhost:8010"
MODEL_FILE = Path("infra/triton/model_repository/e5/1/model.onnx")
SENTENCES = [
    "query: आज मौसम कैसा है?",
    "query: mujhe kal subah saat baje yaad dilana",
    "passage: I'm vegetarian and I don't eat eggs.",
    "passage: मेरी मीटिंग शाम पाँच बजे है।",
]


def get(path: str) -> int:
    try:
        return urllib.request.urlopen(URL + path, timeout=3).status
    except urllib.error.HTTPError as e:
        return e.code
    except urllib.error.URLError:
        return 0  # server not reachable


def main() -> None:
    server, model = get("/v2/health/ready"), get("/v2/models/e5/ready")
    print(f"server ready: {server == 200} ({server}) | model e5 ready: {model == 200} ({model})")
    if model != 200:
        sys.exit("FAIL: model e5 is not ready (check the Triton log for the config error)")

    tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    enc = tok(SENTENCES, padding=True, return_tensors="np")
    body = {
        "inputs": [
            {"name": k, "shape": list(enc[k].shape), "datatype": "INT64", "data": enc[k].flatten().tolist()}
            for k in ("input_ids", "attention_mask")
        ],
        "outputs": [{"name": "sentence_embedding"}],
    }
    req = urllib.request.Request(
        URL + "/v2/models/e5/infer", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t = time.perf_counter()
    out = json.load(urllib.request.urlopen(req, timeout=30))["outputs"][0]
    ms = (time.perf_counter() - t) * 1000
    tri = np.array(out["data"], dtype=np.float32).reshape(out["shape"])
    print(f"triton output: shape {out['shape']}, {out['datatype']}, {ms:.0f} ms (HTTP JSON, cold-ish)")

    sess = ort.InferenceSession(str(MODEL_FILE), providers=["CPUExecutionProvider"])
    ref = sess.run(["sentence_embedding"], {k: enc[k] for k in ("input_ids", "attention_mask")})[0]
    diff, cos = np.abs(tri - ref).max(), (tri * ref).sum(-1).min()
    print(f"parity vs local ORT: max |diff| {diff:.1e}, min cosine {cos:.6f}")
    if diff > 1e-4:
        sys.exit("FAIL: Triton's output doesn't match ORT")
    print("PASS")


if __name__ == "__main__":
    main()
