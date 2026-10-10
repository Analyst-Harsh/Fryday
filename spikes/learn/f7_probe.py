# /// script
# requires-python = ">=3.12"
# ///
"""F session 7 toy: call mlx_lm.server through the OpenAI contract and time it.

Start the server first:
  uv run --with mlx-lm mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8080
Run: uv run spikes/learn/f7_probe.py      (stdlib only; throwaway learning code)
"""

import json
import time
import urllib.request

BODY = {
    "model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "messages": [
        {"role": "system", "content": "You are Fryday. Reply in friendly Hinglish."},
        {"role": "user", "content": "Mujhe 5 tips do ki subah jaldi kaise uthein."},
    ],
    "stream": True,  # tokens arrive one by one as Server-Sent Events (SSE)
    "max_tokens": 250,
}

for run in (1, 2, 3):  # run 1 includes warm-up; runs 2-3 are steady state
    req = urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions",
        data=json.dumps(BODY).encode(),
        headers={"Content-Type": "application/json"},
    )
    print(f"\n--- run {run} ---")
    t0 = time.perf_counter()
    ttft, n_tokens, text = None, 0, ""
    with urllib.request.urlopen(req, timeout=60) as resp:
        for raw in resp:  # each SSE line looks like: data: {...json...}
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            delta = json.loads(line[5:])["choices"][0]["delta"].get("content") or ""
            if delta:
                ttft = ttft or time.perf_counter() - t0
                n_tokens += 1
                text += delta
                print(delta, end="", flush=True)  # show each token the moment it arrives
    total = time.perf_counter() - t0
    tps = (n_tokens - 1) / (total - ttft) if ttft and n_tokens > 1 else 0
    print(f"\n[TTFT={ttft:.2f}s  tokens={n_tokens}  decode={tps:.1f} tok/s  total={total:.2f}s]")
