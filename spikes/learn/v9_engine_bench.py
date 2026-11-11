"""v9 spike: vllm-metal vs mlx_lm.server under concurrent users.

N users send a voice turn at the same moment (Fryday's voice system prompt + one question,
the same settings as LLMClient.stream). Per N: TTFT p50/p95 and total output tokens/s.
Each N runs ROUNDS times, so p95 has a few samples. One warm-up request first, so the
first round doesn't pay the cold start.

With vLLM it also reads the server's own view from /metrics: gauges (KV %, running,
waiting) are sampled every 0.25 s for their peak, counters (pre-emptions, prefix cache)
are read before and after each N. At the end, the engine process's memory from macOS
`footprint` (counts Metal buffers; `ps` RSS doesn't).

Whatever server is on :8080 gets measured. Run it once per server:
  uv run python spikes/learn/v9_engine_bench.py vllm-metal
  uv run python spikes/learn/v9_engine_bench.py mlx_lm.server
"""

import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from fryday.config import Settings
from openai import OpenAI

USERS = [1, 2, 4, 8]
ROUNDS = 3
QUESTIONS = [
    "आज मौसम कैसा है?",
    "मुझे कल सुबह सात बजे याद दिलाना।",
    "एक अच्छी किताब बताओ।",
    "चाय बनाने का आसान तरीका बताओ।",
    "तुम कौन हो?",
    "थोड़ा मोटिवेशन दो यार।",
    "दिल्ली से मुंबई कितनी दूर है?",
    "रात को नींद नहीं आती, क्या करूँ?",
]

s = Settings()
# max_retries=0 like LLMClient: a failed request should show up, not be hidden by retries.
client = OpenAI(base_url=s.llm_base_url, api_key="not-needed", timeout=120, max_retries=0)


def turn(question: str) -> tuple[float, int]:
    """One streamed request → (TTFT in s, output tokens). Mirrors LLMClient.stream's call."""
    t0 = time.perf_counter()
    ttft, tokens = None, 0
    chunks = client.chat.completions.create(
        model=s.llm_model,
        messages=[
            {"role": "system", "content": s.voice_system_prompt},
            {"role": "user", "content": question},
        ],
        stream=True,
        temperature=s.llm_temperature,
        top_p=s.llm_top_p,
        max_tokens=s.llm_max_tokens,
        stream_options={"include_usage": True},
        extra_body={"top_k": s.llm_top_k},
    )
    for chunk in chunks:
        if chunk.usage:
            tokens = chunk.usage.completion_tokens
        if ttft is None and chunk.choices and chunk.choices[0].delta.content:
            ttft = time.perf_counter() - t0
    if ttft is None or tokens == 0:
        raise RuntimeError("empty reply")
    return ttft, tokens


def p95(xs: list[float]) -> float:
    return statistics.quantiles(xs, n=20, method="inclusive")[-1]


METRICS_URL = s.llm_base_url.removesuffix("/v1") + "/metrics"
GAUGES = ["kv_cache_usage_perc", "num_requests_running", "num_requests_waiting"]
COUNTERS = ["num_preemptions_total", "prefix_cache_queries_total", "prefix_cache_hits_total"]


def scrape() -> dict[str, float] | None:
    """vLLM's /metrics → {name: value} for the metrics above; None if the server has none."""
    try:
        text = urllib.request.urlopen(METRICS_URL, timeout=2).read().decode()
    except urllib.error.URLError:
        return None  # mlx_lm.server: no /metrics
    out: dict[str, float] = {}
    for name in GAUGES + COUNTERS:
        # A metric can have several label sets (one per engine); sum them.
        vals = re.findall(rf"^vllm:{name}(?:{{[^}}]*}})? ([0-9.e+-]+)$", text, re.M)
        out[name] = sum(float(v) for v in vals)
    return out


class Sampler:
    """Polls /metrics in a thread while a run is going and keeps each gauge's peak.
    ponytail: 0.25 s polling can miss a shorter spike; Prometheus + max_over_time in prod."""

    def __init__(self) -> None:
        self.peak = dict.fromkeys(GAUGES, 0.0)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            if m := scrape():
                for g in GAUGES:
                    self.peak[g] = max(self.peak[g], m[g])
            self._stop.wait(0.25)

    def __enter__(self) -> "Sampler":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()


def engine_memory() -> str:
    """Engine process memory from macOS `footprint` (includes Metal buffers), or ''."""
    pids = subprocess.run(["pgrep", "-f", "VLLM::EngineCore"], capture_output=True, text=True)
    if not pids.stdout.strip():
        return ""
    pid = pids.stdout.split()[0]
    out = subprocess.run(["footprint", "-p", pid], capture_output=True, text=True).stdout
    now = re.search(r"phys_footprint: (.+)", out)
    peak = re.search(r"phys_footprint_peak: (.+)", out)
    return f"engine memory (footprint): now {now and now[1]}, peak {peak and peak[1]}"


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "server"
    turn(QUESTIONS[0])  # warm-up
    has_metrics = scrape() is not None
    print(f"{label} | {s.llm_model} | {ROUNDS} rounds per N")
    head = f"{'N':>2}  {'TTFT p50':>9}  {'TTFT p95':>9}  {'tok/s (all)':>11}  {'tok/reply':>9}"
    if has_metrics:
        head += f"  {'peak KV':>7}  {'peak run':>8}  {'peak wait':>9}"
        head += f"  {'preempt':>7}  {'prefix hit':>10}"
    print(head)
    for n in USERS:
        ttfts: list[float] = []
        tok_s: list[float] = []
        lens: list[int] = []
        before = scrape()
        with Sampler() as sampler:
            for r in range(ROUNDS):
                qs = [QUESTIONS[(r * n + i) % len(QUESTIONS)] for i in range(n)]
                t0 = time.perf_counter()
                with ThreadPoolExecutor(n) as pool:
                    results = list(pool.map(turn, qs))
                wall = time.perf_counter() - t0
                ttfts += [t for t, _ in results]
                lens += [k for _, k in results]
                tok_s.append(sum(k for _, k in results) / wall)
        row = (
            f"{n:>2}  {statistics.median(ttfts):>8.2f}s  {p95(ttfts):>8.2f}s"
            f"  {statistics.mean(tok_s):>11.1f}  {statistics.mean(lens):>9.0f}"
        )
        after = scrape()
        if before and after:
            d = {c: after[c] - before[c] for c in COUNTERS}
            hit = d["prefix_cache_hits_total"] / max(d["prefix_cache_queries_total"], 1)
            pk = sampler.peak
            row += (
                f"  {pk['kv_cache_usage_perc']:>6.1%}  {pk['num_requests_running']:>8.0f}"
                f"  {pk['num_requests_waiting']:>9.0f}  {d['num_preemptions_total']:>7.0f}"
                f"  {hit:>10.1%}"
            )
        print(row)
    if mem := engine_memory():
        print(mem)


if __name__ == "__main__":
    main()
