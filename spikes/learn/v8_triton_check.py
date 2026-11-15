"""v8 check: is Triton serving e5 correctly, alone and as the text-in `embed` ensemble?

1. Server ready; models e5, tokenizer, embed ready (KServe v2 health endpoints).
2. e5 (IDs in): 4 mixed-length sentences tokenized here, parity vs local ORT.
3. embed (text in): the same sentences as text, parity vs local tokenizer + ORT.
4. Timing, 1 sentence (Fryday's case): client tokenizes + calls e5, vs calls embed.
5. Error path: invalid UTF-8 (sent with the binary data extension; JSON can't carry it)
   must come back as this request's 400 error through the ensemble, and the server must
   keep serving the next request.

Needs `make triton` (HTTP on :8010).
  uv run --with transformers --with onnxruntime python spikes/learn/v8_triton_check.py
"""

import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

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
TOL = 1e-4


def get(path: str) -> int:
    try:
        return urllib.request.urlopen(URL + path, timeout=3).status
    except urllib.error.HTTPError as e:
        return e.code
    except urllib.error.URLError:
        return 0  # server not reachable


def infer(model: str, inputs: list[dict[str, Any]]) -> np.ndarray:
    """KServe v2 JSON request → the `sentence_embedding` output as a numpy array."""
    body = {"inputs": inputs, "outputs": [{"name": "sentence_embedding"}]}
    req = urllib.request.Request(
        f"{URL}/v2/models/{model}/infer", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    out = json.load(urllib.request.urlopen(req, timeout=30))["outputs"][0]
    return np.array(out["data"], dtype=np.float32).reshape(out["shape"])


def id_inputs(enc: Any) -> list[dict[str, Any]]:
    return [
        {"name": k, "shape": list(enc[k].shape), "datatype": "INT64", "data": enc[k].flatten().tolist()}
        for k in ("input_ids", "attention_mask")
    ]


def text_inputs(texts: list[str]) -> list[dict[str, Any]]:
    return [{"name": "TEXT", "shape": [len(texts), 1], "datatype": "BYTES", "data": texts}]


def infer_raw_bytes(model: str, raw: bytes) -> tuple[int, str]:
    """Send one TEXT element as raw bytes (KServe binary data extension) → (status, body).

    Body = JSON header, then the tensor bytes. A BYTES element is a 4-byte little-endian
    length followed by the bytes; Inference-Header-Content-Length says where JSON ends.
    """
    payload = len(raw).to_bytes(4, "little") + raw
    header = json.dumps(
        {"inputs": [{"name": "TEXT", "shape": [1, 1], "datatype": "BYTES",
                     "parameters": {"binary_data_size": len(payload)}}]}
    ).encode()
    req = urllib.request.Request(
        f"{URL}/v2/models/{model}/infer",
        header + payload,
        {"Content-Type": "application/octet-stream", "Inference-Header-Content-Length": str(len(header))},
    )
    try:
        r = urllib.request.urlopen(req, timeout=30)
        return r.status, r.read()[:120].decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def check(name: str, got: np.ndarray, ref: np.ndarray) -> None:
    diff, cos = np.abs(got - ref).max(), (got * ref).sum(-1).min()
    print(f"{name}: shape {list(got.shape)} | max |diff| {diff:.1e}, min cosine {cos:.6f}")
    if diff > TOL:
        sys.exit(f"FAIL: {name} doesn't match local ORT")


def median_ms(fn: Any, runs: int = 30) -> float:
    for _ in range(3):  # warm-up
        fn()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def main() -> None:
    status = {p: get(p) for p in ["/v2/health/ready"] + [f"/v2/models/{m}/ready" for m in ("e5", "tokenizer", "embed")]}
    print("ready:", {p.split("/")[-2] if "models" in p else "server": s == 200 for p, s in status.items()})
    if any(s != 200 for s in status.values()):
        sys.exit("FAIL: not everything is ready (see `make triton-logs`)")

    tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    sess = ort.InferenceSession(str(MODEL_FILE), providers=["CPUExecutionProvider"])
    enc = tok(SENTENCES, padding=True, truncation=True, max_length=512, return_tensors="np")
    ref = sess.run(["sentence_embedding"], {k: enc[k] for k in ("input_ids", "attention_mask")})[0]

    check("e5    (IDs in) ", infer("e5", id_inputs(enc)), ref)
    check("embed (text in)", infer("embed", text_inputs(SENTENCES)), ref)

    one = SENTENCES[:1]

    def client_tokenizes() -> None:
        infer("e5", id_inputs(tok(one, padding=True, truncation=True, max_length=512, return_tensors="np")))

    t_e5 = median_ms(client_tokenizes)
    t_embed = median_ms(lambda: infer("embed", text_inputs(one)))
    print(f"1 sentence, median of 30 (HTTP JSON): client tokenizes + e5 {t_e5:.1f} ms | embed {t_embed:.1f} ms")

    ok_status, _ = infer_raw_bytes("embed", "query: hi there".encode())
    bad_status, bad_body = infer_raw_bytes("embed", b"query: \xff\xfe broken")
    print(f"error path: valid bytes → {ok_status} | invalid UTF-8 → {bad_status}: {bad_body}")
    if ok_status != 200 or bad_status != 400 or "tokenizer:" not in bad_body:
        sys.exit("FAIL: the tokenizer's error didn't come back as this request's error")
    check("after the error ", infer("embed", text_inputs(SENTENCES)), ref)
    print("PASS")


if __name__ == "__main__":
    main()
