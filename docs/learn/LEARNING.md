# Learning journal

This is an append-only log, newest entry first. Write one entry per session (it takes about 5 minutes).

## Mental map
Redraw this weekly in your own words. Each box is filled in by the versions listed under it.

```mermaid
flowchart LR
    C["Client<br/>mic, playback, approval card<br/><i>v4 CLI, v5 browser</i>"] -->|WebSocket| G["App: gateway + turn manager<br/><i>v5</i>"]
    G --> ASR["ASR<br/><i>v4, v8 Triton, v13 LoRA, v14 TRT</i>"]
    G --> A["Agent<br/>tools, approvals<br/><i>v1, v6</i>"]
    A --> LLM["LLM server<br/>OpenAI API<br/><i>v0 MLX, v9 vLLM, v10 quant, v11/v13 LoRA</i>"]
    A --> M["Memory<br/>embed + pgvector<br/><i>v2, v8, v14 TRT</i>"]
    A --> T["Tools / MCP<br/><i>v1, v6</i>"]
    G --> TTS["TTS + normaliser<br/><i>v3, v8</i>"]
    M --> DB[("Postgres<br/>memories, approvals, audit<br/><i>v1, v2, v6, v7b</i>")]
    A --> DB
    E["Eval + observability<br/><i>v7a, v7b</i>"] -.-> LLM
    E -.-> ASR
```

## Still don't get (re-ask after 1 week and again after 4 weeks)
| Added | Question | Re-asked (1 week) | Re-asked (4 weeks) |
|---|---|---|---|
| 2026-10-08 | What is a model **contract** (app ↔ model-server API shape, so backends can be swapped), and how is it different from **structured output** (the model's text following a schema)? | due 2026-10-15 | due 2026-11-05 |
| 2026-10-09 | (F2 quiz, skipped; ask at the F checkpoint) What are query, key and value? Why does a low temperature make output predictable? Why does an LLM need attention instead of fixed word embeddings? | due F session 10 | — |
| 2026-10-13 | (F8 quiz, not attempted; answers were shown instead) (1) Explain the voice → Whisper journey: sampling, resampling to 16 kHz, spectrogram, mel, log, and why each step is needed. (2) What happens if 48 kHz audio is sent to Whisper labelled as 16 kHz? (3) What goes wrong with `x[::3]` and no filter, and where does a 12 kHz whistle land? | due 2026-10-20 | due 2026-11-10 |

## Entry template
```markdown
### YYYY-MM-DD: <version> session N (learn | build | interview)
- **Built or watched:**
- **Predicted vs actual:**
- **Surprised me:**
- **I can now explain:**
- **Still don't get:**
- **Tomorrow's first step:**
```

---

### 2026-10-13: F session 8 (learn): sound and speech overview
- **Built or watched:**
  - Velardo, *Understanding Audio Signals for ML* and *Mel Spectrograms Explained Easily*; 3Blue1Brown, *Fourier Transform*.
  - `spikes/learn/f8_audio.py`: recorded my own voice at 48 kHz; printed samples, PCM16, dBFS and energy by band; resampled to 16 and 8 kHz; plotted the waveform, the STFT spectrogram and the Whisper-style log-mel (80 × 301); played all three versions back.
  - Walked through every part: channels and frames, float → PCM16 (the same ruler idea as weight quantisation, with 65,536 marks), FFT → power (Parseval), Nyquist and aliasing, filtered resampling vs `x[::3]`, the WAV header, STFT windows, the mel filterbank, log.
- **Predicted vs actual:** predictions were not given this session.
- **Surprised me:**
  - Two waveforms with the same tones but different phases are uncorrelated (−0.03), yet their spectra are identical.
  - A linear spectrogram has *more* numbers per second than the waveform (20,100 vs 16,000); the mel is what compresses it (8,000).
  - 2× louder audio shifts every log-mel cell by exactly +6.02 dB.
  - Naive `x[::3]` turned a 10 kHz tone into a fake 6 kHz tone; `librosa.resample` (soxr_hq) removed it.
- **I can now explain:**
  - Sample rate, bit depth, mono/stereo frames.
  - Why energy = amplitude².
  - Nyquist (SR/2) and aliasing.
  - Why resampling filters first.
  - Why ASR uses log-mel: drop phase, keep formants, mel resolution where speech information lives, log for volume invariance. Whisper uses 80 mels (128 in large-v3), 25 ms windows and a 10 ms hop.
- **Still don't get:** quiz not attempted; logged above for re-asking.
- **Tomorrow's first step:** v3 learn 1, the TTS pipeline (text → phonemes → acoustic model → vocoder, RTF), with HF Audio Course ch.6.

### 2026-10-11: v0 build 4, part 2: quantising the real model and measuring it
- **Built:**
  - `mlx_lm.convert` of the original bf16 to 8, 6 and 4 bits (4–6 s each). Our 4-bit is bit-identical to mlx-community's (904/904 tensors).
  - `spikes/learn/v0_quant.py`: memory, TTFT, decode speed, exact match and teacher-forced agreement. The results are in `EXPERIMENTS.md`.
- **Results:**
  - Decode: 15.9 → 29.4 → 36.9 → 53.7 tok/s, 79–84% of the bytes ceiling for every variant.
  - TTFT: 206 → 133 ms.
  - Teacher-forced agreement: 98.5% (bf16 noise floor) / 97.3% / 96.0% / 77.6%.
- **Predicted vs actual:** Claude guessed 4-bit agreement would be 90%+. It was **77.6%**. There is a real step between 6 and 4 bits.
- **Surprised me:**
  - The 4-bit replies read fine by eye even though 0/8 matched bf16 exactly.
  - bf16 against itself is only 98.5%: a batched pass and incremental decoding round differently.
  - Converting took seconds; it's cheap work.
- **I can now explain:**
  - Why fewer bytes means faster decode: the GPU waits on the memory pipe about 95% of the time, and un-quantising is nearly free.
  - Why TTFT improves less: prefill shares one weight read across about 40 tokens, which is the same idea as batching.
  - Divergence vs degradation.
  - Verifying a third-party conversion by rebuilding it from the official source.
- **Still don't get:** —
- **Next:** v0 closed (PR #5; skipped items are listed in the roadmap). The plan was re-ordered to voice first. **Tomorrow's first step: F8, sound and speech overview** (HF Audio Course ch.1), then the `f8_audio.py` toy with your own voice.

### 2026-10-11: v0 build 4, part 1: precision and quantisation
- **Built or watched:**
  - Hugging Face model repo anatomy: config, safetensors, index, tokenizer files, chat template, generation config.
  - Qwen3-4B architecture: about 101 M parameters per layer × 36 layers, plus 0.39 B for the tied embedding, ≈ 4.02 B.
  - Precision formats: fp32, bf16, fp16, int8, int4.
  - Group quantisation (one ruler per 64 weights); "4-bit" really means 4.5 bits per weight.
  - `spikes/learn/v0_quant_toy.py`: quantised layer-0 `q_proj` of the **original** bf16 Qwen by hand.
- **Results:**
  - Error vs bits (group 64): 8-bit 0.57%, 6-bit 2.29%, 4-bit 9.64%, 3-bit 20.7%, 2-bit 48.5%.
  - Group size at 4-bit: 32 → 8.2%, 64 → 9.6%, 128 → 10.9%, whole row → 16.0%.
  - Our formula ≈ `mx.quantize` (9.64% vs 9.82%). The difference is **not** bf16 scales (they are float32 here). MLX anchors the ruler on the edge furthest from 0, then shrinks the gap so that 0 lands exactly on a mark (true in 100% of groups). That clips the other edge, giving a max difference of 0.05.
  - Outlier demo (fixed): a group with one spike (max/median 19×) vs a typical group (3.9×). The other 63 weights: 25.7% vs 9.9% error, using 7 vs 14 of the 16 marks.
  - Adding a bit beats shrinking groups: 4-bit g32 (5.00 bits/weight) 8.2% vs 5-bit g128 (5.25 bits/weight) 5.3%.
  - MLX defaults: affine, 4 bits, group 64. The mlx-community 4/6/8-bit models are all group 64. Other modes: mxfp4 (g32), nvfp4 (g16), mxfp8 (g32), with no bias.
- **Surprised me:**
  - The outlier demo first failed. It picked the widest-range group, whose weights were all large (median 10× typical), so its relative error looked normal. The fix is to select by max/median.
  - "Weights sharing a mark" was meaningless: 64 weights on 16 marks always gives ≥ 48.
  - The handoff note's bf16 guess for the MLX difference was wrong. Checking it found the real cause.
  - The matrix's global min and max sit in the same group of 64, so outliers cluster in columns.
  - Layer anatomy: attention is only ~26% of a layer, and the MLP ~74%. GQA makes K and V 4× smaller than Q.
- **I can now explain:**
  - Why the Instruct model and not the Thinking one (latency).
  - Why Hinglish is weak (incidental in pretraining).
  - Decode is memory-bandwidth-bound: every token streams all weights from unified memory to the GPU cores. The ceiling is tok/s ≈ 153 GB/s ÷ model bytes.
  - mlx-community vs the original (format vs provenance).
- **Still don't get:** —
- **Next:** step 3, `mlx_lm.convert` the original to 8, 6 and 4 bits (group 64) and diff our 4-bit against mlx-community's; then step 4, a full-model check of memory, TTFT, tok/s and agreement.

### 2026-10-10: v0 build 2, part C: tracing with Langfuse
- **Built or watched:**
  - Replaced the hand-written OTel spans with the Langfuse SDK, kept behind reusable helpers in `tracing.py` (the only langfuse import).
  - One trace per turn, a session per chat run, an `llm.stream` generation with TTFT, exact usage and opt-in content.
  - `trace_id` on log lines.
- **Predicted vs actual:**
  - Early stop: I predicted ERROR. Actual: UNSET / `stopped_early`, because `GeneratorExit` isn't a failure.
  - Disabled-client warning: our first guess at the cause was wrong; we verified before fixing.
- **Surprised me:**
  - An infinite recursion: the log processor asked Langfuse for the trace id, and that lookup itself logs.
  - Tests passed alone but failed together, because the root logger leaked between tests.
  - Langfuse warns about missing keys even when tracing is off.
- **I can now explain:**
  - Logs, traces and metrics, and how each is shipped in production (agent / OTLP push / Prometheus scrape).
  - Auto vs manual instrumentation.
  - Vendor code at the seams.
  - The `content()` privacy gate.
  - `env_prefix` vs `validation_alias`.
  - Root dev deps vs app runtime deps.
  - `Generator` vs `Iterator` return types.
- **Still don't get:** —
- **Next:** v0 build 3, my sampler (learner-writes). The live Langfuse check is deferred until needed.

### 2026-10-10: v0 build 2, part A: contract tests with a fake network
- **Built or watched:**
  - `LLMClient(http_client=...)` (dependency injection).
  - `httpx2.MockTransport` fake server.
  - 4 contract tests: deltas joined, request body including `top_k`, mid-stream disconnect and stall, timeout before reply.
  - `conftest.py` isolates Settings from `.env` and `FRYDAY_*` vars.
- **Predicted vs actual:** I predicted a mid-stream disconnect would raise a **raw httpx error and crash** the CLI. ❌ In fact the SDK wraps it: `RemoteProtocolError` → `openai.APIConnectionError`, and `ReadTimeout` → `APITimeoutError`, with the original kept as `__cause__`. `cli.py`'s `except openai.APIError` was already right.
- **Surprised me:**
  - openai 3.x uses **httpx2**, not httpx. Check the library, don't assume.
  - The test answered in 0.25 s what we'd both guessed wrong.
- **I can now explain:**
  - Mocking the SDK vs faking the transport (the real SDK still runs) vs the real server.
  - Dependency injection.
  - TDD: write the test for the desired behaviour, run it, fix if red.
  - `pytest.raises` wrapping one statement.
  - `autouse` fixtures for test isolation.
  - Direct vs transitive dependencies.
- **Still don't get:** —
- **Next:** v0 build 2, part C: OpenTelemetry spans. Part B (the real-server `top_k` check) is pending.

### 2026-10-10: v0 walkthrough of build 1 (config, llm, logs, cli, tests)
- **Built or watched:**
  - Walked through every v0 file slowly.
  - Switched logging to structlog (my call).
  - Added config range checks (fail fast at startup).
  - Corrected the `max_retries` rationale.
- **Predicted vs actual:** I thought `llm.stream()` itself raises when the server is down. In fact a generator doesn't run until it's iterated, so the error appears in the `for` loop.
- **Surprised me:**
  - The `openai` SDK is just an HTTP client. `top_k` goes through `extra_body` to the *server's* sampler, never to the model.
  - "Prompt processing 59/60, 60/60" is prefill: the bulk of the prompt, then the last token on its own, which yields the first reply token.
  - We may not catch a mid-stream disconnect (it could be a raw httpx error). Build 2 will test it.
- **I can now explain:**
  - Config priority: code > env var > `.env` > default. `SecretStr`. Fail fast.
  - The seam and connection reuse.
  - Generators being lazy.
  - The structlog processor pipeline, `foreign_pre_chain` and contextvars (safe for concurrent turns in v5).
  - `perf_counter` (durations) vs `time.time` (timestamps).
  - `trim_history` edge cases.
  - `parametrize`, `monkeypatch`, `capsys`, test isolation.
- **Still don't get:** —
- **Next:** v0 build 2, contract tests with a fake OpenAI server, then OTel spans.

### 2026-10-10: v0 learn 1 + build 1: logits, decoding, first Fryday chat
- **Built or watched:**
  - `v0_logits.py`: real Qwen logits, shape (1, 38, 151936).
  - Temperature, top-k and top-p on real numbers.
  - Built `fryday-chat`: config, the `LLMClient` seam, JSON logs, history trimming, error handling.
- **Predicted vs actual:** —
- **Surprised me:**
  - At temperature 1.5, top-p 0.9 keeps **894** tokens (vs 4 at 0.3).
  - The first live chat hit a repetition loop (484 chunks), and I couldn't reproduce it in 18 runs.
  - The model mixes up "mera" (whose name?): it answered "Mera naam Fryday".
- **I can now explain:**
  - Logits vs probabilities (softmax only cares about differences).
  - Greedy, temperature, top-k, top-p, beam search, logprobs.
  - Why temperature 0 can still vary (near-ties plus floating point).
  - Low temperature for tool calls.
  - Why `max_retries=0` for streaming.
  - Why history keeps the system prompt first (prefix cache).
- **Still don't get:** —
- **Next:** v0 build 2, a contract test against a stub server plus OpenTelemetry spans.

### 2026-10-10: F session 7 (learn): what model hosting means
- **Built or watched:**
  - Hosting layers: weights, runtime, server, contract, app.
  - Weight formats.
  - Memory math (params × bytes).
  - Ran Qwen3-4B 4-bit with `mlx_lm.generate` and with `mlx_lm.server` plus `f7_probe.py`.
  - Also: a PyTorch primer (`f4_pytorch_primer.py`), plus walkthroughs of (B, T, C), nn.Linear and one attention head by hand.
- **Predicted vs actual:**

  | | Predicted | Actual |
  |---|---|---|
  | TTFT | 1–2 s | 0.53 s cold, 0.05–0.11 s warm |
  | Speed | 10 tok/s | about 42–55 tok/s |
  | RAM | about 4 GB | about 2.4–2.6 GB, which matches 4B × 0.5 bytes plus scales |

- **Surprised me:**
  - Speed was 4–5× faster than I guessed.
  - The prompt cache made a repeated prompt's TTFT about 10× smaller.
  - The model claimed it had set a reminder when it had no tool.
- **I can now explain:**
  - What each hosting layer does.
  - Why the contract lets the Mac (MLX) and the GPU (vLLM) swap.
  - Why 4-bit is about 2 GB.
  - Why warm-up exists.
- **Still don't get:** —
- **Next:** short F checkpoint, then v0 (LLM hosting I: build).

### 2026-10-10: F session 3 (learn): tokenisation
- **Built or watched:** Karpathy, *Let's build the GPT Tokenizer*. Toy: `spikes/learn/f3_tokens.py` on Qwen's real tokenizer.
- **Predicted vs actual:** I predicted English < Roman Hinglish < Devanagari ✅. The actual counts were 8 / 12 / 30 tokens.
- **Surprised me:**
  - Devanagari costs 3.75× English.
  - Byte-level BPE splits single Devanagari characters across tokens (the `�` pieces).
- **I can now explain:**
  - BPE sits between characters and words.
  - Byte-level BPE covers every language, at a cost for non-English scripts.
  - More tokens means more latency and more KV-cache memory.
- **Still don't get:** —
- **Tomorrow's first step:** F session 4, building a tiny GPT from scratch (part 1/2).

### 2026-10-09: F session 2 (learn): neural nets → transformers
- **Built or watched:** 3Blue1Brown neural nets, *Transformers* and *Attention*. Toy: `spikes/learn/f2_attention.py`, one attention head by hand.
- **Predicted vs actual:** skipped.
- **Surprised me:**
  - "bank" moves toward water or money purely from its neighbours.
  - The weights come out flat without the learned W_q and W_k.
- **I can now explain:**
  - Attention as three steps: score (q·k), normalise (softmax), blend (V).
  - Temperature sharpens or flattens the next-token distribution.
  - Past tokens' K and V never change, so they're cached (the KV cache).
- **Still don't get:** quiz deferred (see the table).
- **Tomorrow's first step:** F session 3, Karpathy's tokenizer lecture, plus a Hinglish token-count toy.

### 2026-10-08: F session 1 (learn): the big picture
- **Built or watched:** Karpathy, *Intro to Large Language Models*. Walked through one Fryday turn (architecture doc §1 and §4).
- **Predicted vs actual:** I predicted "model processing" eats most of the ~2 s. More precisely, ASR is the biggest cost on the Mac (1–2 s), and the LLM's time to its first chunk matters most on the GPU. Streaming hides the rest.
- **Surprised me:**
  - The app is a **hub**, not a pipeline: models never call each other.
  - The LLM can be called twice in one turn (once to decide on a tool, once to phrase the reply).
- **I can now explain:**
  - An LLM is two files, weights plus a run program, and the hosting layer (MLX, vLLM, Triton) is that run program.
  - The LLM only *proposes* tool calls. The app validates them and executes them.
  - Pretraining gives the model knowledge. Fine-tuning gives it behaviour and format.
  - Streaming overlaps LLM and TTS.
- **Still don't get:** contract vs structured output (I mixed them up in the quiz; see the table above).
- **Tomorrow's first step:** F session 2, the 3Blue1Brown neural-network refresher, then *Transformers* and *Attention*.
