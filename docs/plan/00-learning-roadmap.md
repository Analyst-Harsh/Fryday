# Fryday — Learning Roadmap

| | |
|---|---|
| **Supersedes** | the *sequencing* of [00-roadmap.md](00-roadmap.md). The phase files stay as the deep-detail backlog; the [HLD](../specs/2026-10-04-fryday-hld-design.md) stays the north star. |
| **Date** | 2026-10-07 |
| **Pace** | ~1.5 h sessions, 5 days/week (default) |
| **Learning files** | [resources](../learn/resources.md) · [journal](../learn/LEARNING.md) · [production gaps](../learn/production-gaps.md) · [interview kit](../learn/interview-kit.md) |

## Why this exists
The 15-phase roadmap is ordered by risk retirement, which makes it a good architecture plan and a poor curriculum. Phase 1 (spike 0) asked for nine experiments before a single token had flowed through the system. This roadmap re-orders the same destination into **runnable versions**. Each version follows the same loop:

**learn → see where it fits → build it (production-grade) → demo → explain it back.**

The goal is to be interview-ready for AI engineering roles while building Fryday.

## Definition of done (every version)
- [ ] The demo in the version table runs, and a gif or screenshot is in the PR.
- [ ] Every learner-writes item for this version (§5) is written by the owner.
- [ ] `EXPERIMENTS.md` has an entry for every measurement or spike answered, with model revision SHAs.
- [ ] `docs/learn/vN.md` holds the owner's explain-it-back note.
- [ ] The interview session (I) is done, and weak answers are logged in `LEARNING.md`.
- [ ] New shortcuts are added to `production-gaps.md`, and the HLD is updated if a decision changed.
- [ ] CI is green and one PR is merged per version.

## Risks
| Risk | Mitigation |
|---|---|
| Rework when v8 swaps backends | Seam rule: every model sits behind one typed function or Protocol with a contract test from v0 |
| Hindi TTS quality stalls v3 | 2-session timebox; fall back to MMS, or English audio plus Devanagari text on screen |
| A fun assistant at v6 absorbs all the time, and the deep topics never happen | Checkpoint after v6: Part B is mandatory |
| GPU cost | $100 cap, learning and dry runs before renting, auto `gpu-down`, drop-first stretch list |
| Young tools (vllm-metal, Triton arm64, MLX DPO) | 1-session timebox each, with the written fallback in the version |

## 1. Shape: 5 parts, 19 runnable steps

Sessions are about 1.5 h. "L" is learn sessions, "B" is build sessions, and "I" is the interview session that closes each version.

### Part 0: Foundations (Mac)
| # | Version | L+B+I | Demo |
|---|---|---|---|
| F | **Architecture + core theory** (§3), including a tiny GPT built from scratch | 10 | You explain one Fryday turn on your own diagram, and your own mini-GPT generates text |

### Part A: Build the assistant (Mac)
| # | Version | L+B+I | Demo |
|---|---|---|---|
| v0 | LLM hosting I: local LLM behind an OpenAI API, streaming, sampling from scratch, quantisation taste | 2+4+1 | Type Hinglish, tokens stream back |
| v1 | Tool calling, structured output, agent loop, MCP | 2+5+1 | "Remind me at 6…" works |
| v2 | RAG and memory: embeddings, hybrid search, reranking, RAG metrics; ONNX intro | 2+6+1 | "I'm vegetarian" is recalled the next day |
| v3 | Speech I: audio basics + TTS + text normalisation | 2+4+1 | Fryday speaks |
| v4 | Speech II: ASR, WER, Hinglish script decision, recording your own-voice test set | 2+5+1 | Speak, get a reply |
| v5 | Real-time streaming turn: WebSocket, barge-in, VAD endpointing, latency | 3+9+1 | **Hero video:** talk to it in the browser |
| v6 | Reliability and safety: the money path | 3+12+1 | "Order milk" gives exactly 1 order under every fault |

### Part B: Model engineering (Mac): the core interview depth
| # | Version | L+B+I | Demo |
|---|---|---|---|
| v7a | Evaluation: frozen sets, judge, gate, `models.lock` | 2+5+1 | `make eval` prints a score table |
| v7b | Observability + operations: Langfuse, SLOs, backups, supply chain | 2+5+1 | Traces visible; an injected alert fires |
| v8 | Model serving: ONNX → Triton | 2+7+1 | The app runs with models behind Triton (or ORT) |
| v9 | LLM serving internals + vLLM | 3+4+1 | Goodput and TTFT vs concurrency curve |
| v10 | Quantisation | 2+4+1 | Quality, size and speed table |
| v11 | Fine-tuning I: data + LoRA + DPO on the Mac | 3+9+1 | Before/after on the eval |

### Part C: GPU (rented sprints; learning and dry runs happen before the meter runs)
| # | Version | L+B+I | Demo |
|---|---|---|---|
| v12 | CUDA/GPU fundamentals (you write a kernel) + training-systems concepts + GPU ops | 3+5+1 | Your kernel vs cuBLAS; tunnel up; dead-man switch tested |
| v13 | Fine-tuning II + quantisation at scale + vLLM on CUDA + multi-LoRA | 2+8+1 | Promoted artefacts in `models.lock` |
| v14 | TensorRT + Triton on GPU + profiling | 2+9+1 | PyTorch vs ONNX vs TRT table, Nsight timeline |
| v15 | Prove it: load, cost, chaos, demo | 1+5+1 | Recorded demo + benchmark README |
| v16 | Interview capstone (§6) | 6 | 3 mock system-design rounds, resume bullets |

**Total:** about **177 sessions**. The default pace is 1.5 h a day, **5 days a week**, which is about 8 months, or about 10 with slip. At 6 days a week it is about 7 months. Along the way:
- a text assistant runs at about **week 5**;
- the voice hero video lands at about **week 13**;
- Part B runs from about week 16 to week 26.

**Checkpoint after v6:** Part B is mandatory, because it holds most of the interview depth.

**GPU budget:** a hard cap of **$100** (you can change it in v12). That is roughly 60–100 GPU-hours on a 24 GB L4-class card, at about $0.5–1/h; verify the rates in v12. Every sprint has a written hour limit, and `gpu-down` fires automatically at that limit. If the cap would be exceeded, drop stretch items, never the checklist.

**Drop-first stretch items** run only if the sprint finishes under budget:
- E5 (TRT-LLM Whisper encoder);
- E7 (one Triton vs separate servers);
- speculative decoding;
- DPO at scale;
- Nemotron streaming ASR.

**Learn spill rule:** if a version's learn sessions run short for its topic load, they may spill one session into the build block. The first build session of v5, v6 and v12 starts with a 20-minute toy.

**Setup items, scheduled when first needed:**

| Version | Setup items |
|---|---|
| v0 | Hugging Face token |
| v4 | Dataset terms and downloads: MUCS (OpenSLR 104), Kathbath, IndicVoices; skip CS-FLEURS |
| v6 | Google Cloud OAuth and web search key |
| v7a | Hosted LLM and ASR baseline keys |
| v7b | Langfuse (EU) |
| v12 | GPU provider plus budget alert |

**Why this order:**
- Build the product first so every later concept has a home.
- Measure (v7a) before you change any model (v8–v11).
- Learn serving on the Mac (v8, v9) before paying for a GPU.
- Do fine-tuning and quantisation on the Mac first, then repeat at scale on the GPU.

**Spike 0 is folded into the versions:**

| Spike | Answered in |
|---|---|
| S0-3 | v0, v1 |
| S0-5 | v0, v2, v3, v4 |
| S0-9 | v3 |
| S0-8 | v3, v4 |
| S0-2 | v5 |
| S0-10 | v6 |
| S0-1 | v8 |
| S0-6, 7, 11 | v12 |
| S0-4 (real Swiggy access) | Stays deferred, as in the HLD |

**HLD milestones → versions:**

| Milestone | Versions |
|---|---|
| M0 | Spikes, as above |
| M1 | v1, v2, v6, v7a |
| M2 | v3, v4, v5 |
| M3 | v10, v11, v13 |
| M4 | v8, v14 |
| M5 | v11, v13, v14 |
| M6 | v15 |

---

## 2. Production standard

### Production baseline (each row applies from the version it names, or v0 if none; none is deferred past it)
| Practice | Why it matters |
|---|---|
| Typed config (`pydantic-settings`); secrets only in env; gitleaks (already in place) | Twelve-factor config, no leaked keys |
| Structured JSON logs with `turn_id` / `request_id` | Debuggable when there is no debugger attached |
| OpenTelemetry spans around every model and tool call (console exporter until v7b) | Latency attribution, the core voice-AI skill |
| An explicit **timeout** on every network or model call; retries only when the call is idempotent | Fail fast, never double-act |
| Typed seams: every model behind one function or Protocol, with a contract test against a stub | Swap backends without touching callers (this is what v8 relies on) |
| Tests for all logic; CI on every PR (in place); ruff and pyright strict | Regression safety |
| Pinned dependencies (`uv.lock`); every model pinned by HF revision SHA from v0 | Reproducibility, which becomes `models.lock` in v7a |
| Health and readiness endpoints once a service exists (v5) | Orchestration-ready |
| Decisions recorded: `EXPERIMENTS.md` (numbers), HLD updates (design) | ADR habit |
| Small PRs, one branch per version, conventional commits | Reviewable history |
| Structured error taxonomy (user-facing vs retryable vs fatal); forward-only migrations with a written rollback note | Predictable failure behaviour |
| Prompts versioned in git; prompt version logged on every trace | Debug "what changed?" |
| Concurrency limit and load shedding at the gateway (v5); a circuit breaker plus a degrade path per backend (HLD §9) | Survive overload and dead backends |
| Input and output guardrails: injection suite (v6), PII filter (v7b), refusal/jailbreak test set (v7a) | Safety you can show |
| Container hygiene: non-root, multi-stage builds, digest-pinned images | Supply chain, attack surface |
| Warm-up and cold-start time measured per model (v8) | Readiness honesty |

### Deliberate production gaps (`docs/learn/production-gaps.md`)
Every shortcut is written down with four things: **what production teams do**, **why we skip it now**, **when it would matter**, and **the interview talking point**. The starting register:

| Production standard | Our choice | Why, for now |
|---|---|---|
| WebRTC (+ TURN) for real-time voice | WebSocket PCM | Simpler and single-user; you should be able to explain the jitter, NAT and UDP trade-offs |
| Kubernetes, autoscaling on GPU metrics, sticky WebSocket load balancing | docker compose + scripts | Single user, one box |
| Terraform / IaC | Provider-CLI scripts | One provider, short-lived boxes |
| Secrets manager (Vault/SSM), KMS | `.env` + Fernet | Local single user |
| Multi-tenant auth, OAuth login, per-user rate limits | Static token, localhost only, global daily minute cap (v5) | Single user |
| Canary / blue-green model rollout, shadow traffic | Eval gate + `models.lock` rollback | No real traffic to split |
| Model registry service, feature store | MLflow local file store, git | Solo scale |
| Managed Postgres, PITR, encryption at rest | Local Postgres, nightly encrypted dump | Learning scale |
| SBOM + image signing (cosign), SLSA | Digest pinning, pip-audit, Trivy | Cheap subset first |
| On-call paging (PagerDuty) | Desktop notification | Solo |
| Multi-GPU: tensor/pipeline parallelism | One 24 GB GPU | A 4B model fits; you learn this as a concept only |
| RLHF with PPO/GRPO and a reward model | SFT LoRA plus a small DPO run (v11) | Cost and data; you should be able to explain PPO vs GRPO vs DPO |
| Speaker verification, wake word, learned end-of-turn model | Tap-to-talk + UI tap; Silero VAD as an option (v5) | Scope (HLD Phase 2) |
| DPDP/GDPR process, DPA with vendors | Retention, `forget`, PII scrubbing | Personal project |
| Streaming ASR (partial transcripts) | Batch ASR on the tapped utterance | Simpler; Nemotron streaming is a v8 stretch |
| Semantic / response caching | None | Low repeat traffic |
| Online eval, A/B tests, drift monitoring, user-feedback loop into training data | Offline frozen-set eval | One user, no traffic |
| Content moderation model | Prompt rules + injection suite | Single trusted user |
| Per-request cost attribution | Cost per minute measured in v15 | Measured once, not live |
| Load balancing across model replicas | One replica per model | Single box |
| DR targets (RPO/RTO), multi-region | Nightly dump, RPO ≈ 24 h | Personal data only |
| Licence compliance process | Licence column in `models.lock` | Manual review suffices |
| Speech-to-speech models (Moshi-style) | Cascaded ASR → LLM → TTS | Explain the trade-off (latency vs control and tool use) |

Each version adds its own rows as they come up.

---

## 3. Version detail
Each version lists what you **learn** (interview topics), what you **build**, the **production** points it adds, and the deep-detail **source** in the old plan.

### F: Foundations (10 sessions)
> **Reordered 2026-10-10 (just-in-time learning).** Sessions 1–3 are done. Next comes **7 (hosting)**, then a short checkpoint, then **v0**. Sessions 4–6 (tiny GPT, modern architecture) move to just before **v9**, where attention internals and the KV cache become things you measure. Sessions 8 and 9 (speech, real-time) move to just before **v3** and **v5**. Nothing is cut. The rule from here on: learn depth when the build needs it.

1. **The big picture.** Karpathy, *Intro to LLMs*. Claude walks you through architecture doc §1 and §4, and you draw your six-box diagram.
2. **Neural nets refresher → transformers.** 3Blue1Brown's neural-network and transformer/attention videos. Covers embeddings, attention, MLP, softmax and residuals.
3. **Tokenisation.** Karpathy's tokenizer lecture: BPE and byte-level BPE. Toy: measure tokens per word for English vs Roman Hinglish vs Devanagari.
4. **Build a tiny GPT from scratch (1/2).** Karpathy, *Let's build GPT*: self-attention in code.
5. **Build a tiny GPT from scratch (2/2).** Train it on the MPS device, make it generate, and inspect the attention weights.
6. **Modern LLM architecture and training.** Karpathy, *Deep Dive into LLMs*. Covers:
   - MHA → MQA/GQA, RoPE, RMSNorm, MoE and long context (sliding window, "lost in the middle");
   - pretrain → SFT → RLHF (PPO/GRPO) vs DPO, reward hacking (concepts).
7. **What "model hosting" means.**
   - Weights and formats: safetensors, GGUF, MLX, ONNX.
   - Runtime vs server vs API contract.
   - Where MLX, vLLM, Triton, ORT and TensorRT sit.
   - Memory math (params × bytes), the 16 GB budget.
   - Toy: `mlx_lm.generate`, then `mlx_lm.server` called with `curl`.
8. **Sound and speech overview.** HF Audio Course ch.1 and the ch.5–6 overviews.
9. **Real-time voice systems.** Pipecat and LiveKit concepts. Cascaded vs speech-to-speech pipelines; the latency budget. Toy: fill in the budget table.
10. **Checkpoint.** You redraw the architecture and take a 15-question quiz. Output: `docs/learn/00-architecture-map.md`.

### v0: LLM hosting I
- **Learn:**
  - the OpenAI chat API and SSE streaming;
  - TTFT vs TPOT vs throughput;
  - quantisation intuition: bf16 vs 8-bit vs 4-bit memory and speed on MLX;
  - prompt basics, system prompts, Hinglish style;
  - decoding: greedy, temperature, top-k, top-p, repetition penalty, beam search, logprobs, and why temperature 0 is still non-deterministic.
- **Build:**
  - a CLI chat against `mlx_lm.server` with streaming, and TTFT and tokens/s measured;
  - **your own sampler** (~50 lines) over raw logits from `mlx_lm`, compared against the server's output.
- **Production:** lay the baseline from §2: config, logging, OTel console spans, timeouts, and a contract test against a stub OpenAI server.
- **Source:** old phases 2 and 3.

### v1: Tool calling
- **Learn:**
  - the function-calling loop;
  - JSON schema;
  - constrained decoding vs validate-and-repair;
  - agent patterns (Anthropic, *Building effective agents*), multi-agent patterns as a concept;
  - loop caps;
  - context engineering (what goes in the window, compaction);
  - **MCP internals** (client/server, tools, resources, transports);
  - agent evals (tool-selection accuracy, trajectory).
- **Build:** a hand-written agent loop; notes and reminders in Postgres with Alembic; Pydantic validation with one repair attempt; S0-3 measured (at least 18 of 20 streamed tool calls valid).
- **Production:** tool timeouts, an idempotency key on non-spend tools, a single retry.
- **Source:** old phases 3 and 4.

### v2: Memory / RAG
- **Learn:**
  - embeddings and cosine similarity;
  - ANN indexes (HNSW vs IVF, recall/latency trade-off);
  - pgvector, chunking;
  - hybrid search (BM25 + dense, reciprocal rank fusion);
  - reranking with cross-encoders;
  - query rewriting;
  - RAG metrics (recall@k, MRR, faithfulness);
  - RAG failure modes;
  - ColBERT and GraphRAG as concepts;
  - memory extraction.
- **ONNX intro:** export the embedder and run a parity test against PyTorch.
- **Build:**
  - a pgvector memory table;
  - hybrid retrieval using Postgres full-text search plus vectors;
  - a small reranker;
  - recall@k measured on 30 labelled queries;
  - the first 30-case tool-call mini eval.
- **Source:** old phase 4.

### v3: Speech I: voice out
- **Learn:**
  - PCM, sample rate, resampling, WAV;
  - TTS pipeline (text → tokens/phonemes → acoustic → vocoder), RTF;
  - text normalisation;
  - TTS quality metrics (MOS, UTMOS-style predictors);
  - codecs (Opus) as a concept.
- **Build:** a `speak(text) -> pcm` seam using Kokoro (`mlx-audio`) or MMS, with a 2-session timebox and a fallback. Kokoro's Hindi quality is unverified, and MMS is CC-BY-NC (fine for learning; record it in `models.lock`). Add the Hindi numbers normaliser with golden tests (S0-9), and listening checks (S0-5, half of S0-8).
- **Source:** old phase 7 and HLD §3.1.

### v4: Speech II: voice in
- **Learn:**
  - log-mel spectrograms;
  - Whisper encoder–decoder vs CTC models, streaming ASR;
  - WER and CER;
  - hallucination on silence;
  - the code-switching / script problem.
- **Build:** a `transcribe(pcm) -> str` seam with `mlx-whisper` and a tap-to-talk CLI. Score Oriserve (Roman) vs Whisper (Devanagari) on MUCS clips with `jiwer`, then **decide the script convention** (S0-8). **Record your 30–60 min own-voice entity test set** (1 session; used later in v7a). Update the HLD if the TTS choice moves away from Magpie.
- **Source:** old phase 7.

### v5: Real-time streaming turn
- **Learn:**
  - asyncio, queues, backpressure, cancellation;
  - WebSocket vs WebRTC;
  - sentence chunking;
  - jitter buffers;
  - barge-in epochs;
  - p50 vs p95 latency;
  - admission control;
  - **VAD, endpointing and turn detection**;
  - echo cancellation (AEC);
  - streaming-ASR chunking as a concept.
- **Build:**
  - a FastAPI WebSocket gateway (token auth in the first message, Origin check, protocol `v` field);
  - a turn manager and a thin JS mic client;
  - per-hop spans;
  - the full memory-fit run (S0-2), with early RSS checks already done in v3 and v4 as each model was added;
  - **Silero VAD endpointing** as an optional hands-free mode (tap-to-talk stays the default), with the end-of-speech threshold measured.
- **Production:**
  - readiness/health endpoints;
  - graceful shutdown;
  - a bounded queue;
  - a concurrency limit and load shedding;
  - a per-user daily minute cap;
  - a circuit breaker with a "GPU offline" degrade message.
- **Source:** old phase 8.

### v6: Reliability and safety (money path)
- **Learn:**
  - state machines;
  - compare-and-set in SQL;
  - idempotency;
  - at-most-once vs exactly-once;
  - reconciliation;
  - LangGraph checkpointer and interrupt;
  - prompt injection and untrusted tool output;
  - OAuth least scope;
  - an append-only audit log.
- **Build:**
  - a mock grocery MCP server with fault injection;
  - the approval state machine and reconciler;
  - migrate the agent to LangGraph;
  - an approval card;
  - an injection test suite;
  - Google Calendar (OAuth, Fernet) and web search.
- **Invariant:** orders per approval ≤ 1 under every fault.
- **Source:** old phases 4 and 5.

### v7a: Evaluation
- **Learn:**
  - frozen test sets and data leakage;
  - LLM-as-judge calibration;
  - confidence intervals and bootstrap;
  - release gates;
  - model provenance.
- **Build:**
  - a 200-case LLM frozen set;
  - a refusal/jailbreak set;
  - ASR frozen sets (MUCS/Kathbath plus the own-voice entity set) and a synthetic dev set;
  - hosted baselines run offline;
  - `eval/gate.yaml`;
  - `models.lock`.
- **Source:** old phase 6.

### v7b: Observability + operations
- **Learn:**
  - tracing vs metrics vs logs;
  - SLIs, SLOs and error budgets;
  - backup and restore;
  - retention;
  - supply chain.
- **Build:**
  - Langfuse with a PII scrubber;
  - Prometheus and an SLO alert script;
  - nightly encrypted `pg_dump` with a restore drill;
  - retention, `forget`, audio opt-in and delete;
  - pip-audit and Trivy;
  - `RUNBOOK.md`.
- **Source:** old phase 9.

### v8: Model serving: ONNX → Triton
- **Learn:**
  - ONNX graphs, opsets, ORT execution providers;
  - KServe v2 protocol;
  - Triton model repo, `config.pbtxt`, backends (ONNX, Python), dynamic batching, instance groups, ensembles, metrics;
  - gRPC streaming and cancellation;
  - warm-up and readiness.
- **Build:**
  - Arm64 CPU Triton (S0-1) gets a **1-session timebox**; if it fails, ORT goes behind the same contract.
  - Embeddings, ASR and **TTS (Python backend)** go behind KServe v2.
    - Docker on the Mac has **no Metal access**, so Triton runs CPU stand-ins: an ONNX embedder, faster-whisper or an ONNX Whisper, and a CPU TTS.
    - The MLX models stay native.
    - Contract tests check schema and latency, not quality parity.
    - Cap the Docker VM at 4 GB or less.
  - Swap the seams; contract tests pass against both.
  - Measure warm-up and cold start.
  - Note in the gaps register that dynamic batching on CPU won't show GPU batching behaviour; that is revisited in v14.
  - Nemotron streaming ASR as a stretch.
- **Source:** old phases 2 and 7, HLD §10.

### v9: LLM serving internals + vLLM
- **Learn:**
  - prefill vs decode (compute-bound vs memory-bound);
  - the KV-cache size formula, and how GQA shrinks it;
  - PagedAttention;
  - continuous batching;
  - chunked prefill;
  - prefix caching;
  - queueing: Little's law, goodput under an SLO;
  - concepts: speculative decoding, FlashAttention (tiling and recompute derived on the roofline), prefill/decode disaggregation, KV offload, multi-LoRA serving.
- **Build:**
  - the KV-cache calculation for Qwen3-4B on paper;
  - vllm-metal vs `mlx_lm.server`, benchmarked for concurrency against TTFT and throughput. vllm-metal is young, so it gets a **1-session timebox**. If it fails, or lacks continuous batching or prefix caching, use the fallback:
    - a paper KV / Little's-law model;
    - a toy continuous-batching scheduler simulator;
    - the real vLLM benchmark moves to v13;
    - the change is logged in `EXPERIMENTS.md`.
  - an OpenAI-contract swap with no app change.
- **Source:** old phases 10 and 14, E9 groundwork.

### v10: Quantisation
- **Learn:**
  - number formats (fp32/bf16/fp16/fp8/int8/int4);
  - symmetric vs asymmetric quantisation, per-tensor vs per-channel vs per-group;
  - PTQ vs QAT;
  - AWQ, GPTQ, GGUF and MLX formats;
  - KV-cache quantisation;
  - distillation and pruning, as concepts.
- **Build:**
  - E3, a from-scratch int8/int4 quantiser on one layer with error analysis;
  - MLX bf16 vs 8-bit vs 4-bit scored on the v7a eval;
  - an ONNX int8 embedder vs fp32.
- **Source:** old phase 10.

### v11: Fine-tuning I
- **Learn:**
  - when to fine-tune vs prompt vs RAG;
  - SFT;
  - LoRA and QLoRA (rank, alpha, target modules), merge vs adapter serving;
  - loss masking on assistant tokens;
  - memory math: full fine-tune (weights + grads + Adam states + activations) vs LoRA;
  - catastrophic forgetting;
  - the DPO loss;
  - synthetic data generation and filtering;
  - dedup and contamination;
  - experiment tracking.
- **Build:**
  - a synthetic tool-call data pipeline validated against the mock tools;
  - DVC and MLflow;
  - MLX LoRA on 4-bit Qwen3-4B, before vs after on the gate. Memory-safe settings: sequence length 512 or less, batch 1, gradient checkpointing, other services stopped. Fallback: Qwen3-1.7B.
  - **DPO:** mlx-lm has no official DPO, and DPO needs a reference model in memory. So **you implement the DPO loss yourself** and run a few steps on Qwen3-0.6B or 1.7B. If that isn't stable after 1 session, the real DPO run moves to v13.
  - ASR training manifests, plus a few-step ASR LoRA dry run;
  - GPU training scripts dry-run on the Mac.
- **Source:** old phase 10.

### v12: CUDA/GPU fundamentals + GPU ops
- **Learn:**
  - GPU architecture: SMs, warps, threads/blocks/grid;
  - the memory hierarchy (HBM, shared, registers) and bandwidth vs FLOPs (the roofline);
  - streams, pinned memory, host↔device copies, kernel launch overhead;
  - coalescing, bank conflicts, occupancy, tensor cores;
  - mixed precision (fp16 loss scaling vs bf16);
  - `nvidia-smi` and DCGM;
  - **training systems (concepts):** DDP, FSDP/ZeRO, gradient checkpointing, gradient accumulation, tensor and pipeline parallelism.
- **Build:**
  - **your own CUDA kernels**, timed against cuBLAS (CUDA C or Triton-lang): vector add, then a tiled matmul. This is a paid-GPU session, scheduled explicitly;
  - a provider choice;
  - `gpu-up` / `gpu-down`;
  - a hardened SSH tunnel;
  - a dead-man switch, a heartbeat alert, a budget alert;
  - the S0-6, S0-7 and S0-11 spikes.
- **Source:** old phase 11.

### v13: Fine-tuning II + quantisation at scale
- **Learn (before renting):**
  - PEFT/TRL on CUDA, QLoRA with bitsandbytes;
  - AWQ vs GPTQ calibration in practice;
  - vLLM CUDA flags (`--gpu-memory-utilization`, max sequences, quantisation, LoRA);
  - the sprint checklist.
- **Build:**
  - the full synthetic data run;
  - PEFT LoRA/QLoRA on CUDA, merged;
  - AWQ and GPTQ;
  - vLLM on CUDA, including **multi-LoRA serving** of adapters vs a merged model;
  - ASR LoRA;
  - gate and promote;
  - E1 (constrained decoding vs fine-tune) and E2 (quantisation formats).
  - Stretch: DPO at scale on the GPU.
- **Source:** old phase 12.

### v14: TensorRT + Triton on GPU + profiling
- **Learn:**
  - TensorRT builder, engines, precision;
  - **INT8 calibration**, the timing cache, plugins;
  - dynamic shapes, layer fusion;
  - CUDA graphs;
  - Nsight Systems.
- **Build:**
  - E4 (PyTorch vs ONNX vs TRT fp32/fp16/int8 on embeddings);
  - E5 stretch (TRT-LLM Whisper encoder);
  - E6 (CUDA graphs);
  - Triton dynamic batching measured on the GPU;
  - E7 stretch (one Triton vs separate servers);
  - E8 (Triton vLLM backend vs standalone);
  - an annotated Nsight timeline;
  - Mac-vs-GPU divergence;
  - speculative decoding on vLLM as a stretch experiment.
- **Source:** old phase 13.

### v15: Prove it
- **Learn:**
  - load-testing methodology (open vs closed loop, coordinated omission);
  - capacity planning;
  - cost modelling;
  - chaos engineering.
- Load test (k6 or Locust): E9 (KV cache and batching vs concurrency), E10 (max concurrency at the SLO), E11 (cost per conversation-minute).
- Set the admission limit from the data.
- Chaos: drop the tunnel mid-spend.
- Record the demo; write the README.
- **Source:** old phase 14.

---

## 4. Topic coverage checklist (verification against interview scope)
Depth: **B** = built, **M** = measured in an experiment, **C** = concept only, with the reason given.

| Area | Topics | Where | Depth |
|---|---|---|---|
| LLM fundamentals | transformer, attention (from scratch), BPE, MHA/GQA, RoPE, RMSNorm, MoE, long context, pretrain → SFT → RLHF/PPO/GRPO vs DPO | F | B (tiny GPT) + C |
| Decoding | greedy, temperature, top-k, top-p, beam, repetition penalty, logprobs, non-determinism | v0 | B (own sampler) |
| LLM hosting | runtimes vs servers, formats, OpenAI API, streaming, memory math, TTFT/TPOT | F, v0, v9 | B, M |
| vLLM | PagedAttention, continuous batching, chunked prefill, prefix cache, multi-LoRA, vllm-metal → CUDA | v9, v13, v14, v15 | B, M (E8, E9) |
| LLM serving advanced | goodput, Little's law, speculative decoding, FlashAttention, disaggregation, KV offload, tensor/pipeline parallelism | v9, v12, v14 | M for goodput and the spec-decoding stretch; the rest is C (parallelism is C because there is one GPU) |
| Triton | model repo, `config.pbtxt`, ONNX and Python backends, dynamic batching, instance groups, ensembles, gRPC, metrics, vLLM backend | v8, v14 | B, M (E7 stretch, E8) |
| ONNX | export, opsets, parity, ORT providers, int8 | v2, v8, v10 | B |
| Quantisation | formats, scale and zero-point, granularity, PTQ/QAT, AWQ, GPTQ, GGUF, MLX, KV quantisation, INT8 calibration; distillation and pruning | v0, v10, v13, v14 | B, M (E2, E3); distillation is C |
| TensorRT | engines, precision, INT8 calibration, timing cache, plugins, dynamic shapes, fusion, TRT-LLM | v14 | B, M (E4, E5 stretch) |
| CUDA / GPU | SMs, warps, memory hierarchy, coalescing, occupancy, tensor cores, roofline, streams, pinned memory, CUDA graphs, Nsight, mixed precision | v12, v14 | B (own kernels) + M (E6, Nsight) |
| Training systems | DDP, FSDP/ZeRO, gradient checkpointing and accumulation, loss scaling | v12 | C |
| Fine-tuning | when to tune, SFT, loss masking, memory math, LoRA/QLoRA, merge vs adapter, data synthesis and hygiene, forgetting, DPO | v11, v13 | B, M (E1) |
| Agents | tool calling, structured output, constrained decoding, MCP, LangGraph, checkpoints, interrupts, context engineering, agent evals; multi-agent | v1, v6, v13 | B, M (E1); multi-agent is C |
| RAG / memory | embeddings, ANN indexes, pgvector, chunking, hybrid search, reranking, query rewriting, recall@k/MRR/faithfulness; ColBERT, GraphRAG | v2 | B; ColBERT and GraphRAG are C |
| Evaluation | frozen sets, leakage, LLM judge, CIs, gates, WER/CER, refusal sets | v2, v4, v7a | B |
| Speech | audio basics, mel, Whisper vs CTC, WER, TTS + vocoder, MOS, normalisation, code-switching, codecs | F, v3, v4 | B |
| Real-time systems | streaming, backpressure, cancellation, barge-in, VAD/endpointing, AEC, WebSocket vs WebRTC, p95; speech-to-speech | v5 | B; WebRTC and speech-to-speech are C (see gaps) |
| Safety | prompt injection, guardrails, approvals, idempotency, at-most-once, audit, PII | v6, v7a, v7b | B |
| MLOps | DVC, MLflow, `models.lock`, provenance, reproducibility, rollback | v7a, v11, v13 | B |
| Ops / SRE | SLOs, alerts, tracing, backups, retention, supply chain, runbooks, circuit breakers, load shedding, IaC-lite, cost | v5, v7b, v12, v15 | B; gaps registered |
| System design | voice assistant at scale, LLM serving at scale, RAG platform, capacity planning, build vs buy | v16 | Mock interviews |

**Completeness check (done while planning):**
- every row of the HLD §4 learning map maps to a version;
- milestones M0–M6 map;
- experiments E1–E11 map;
- spikes S0-1…S0-11 map;
- old phases 2–14 map (phase 6 → v7a, phase 9 → v7b);
- the old roadmap's coverage-map rows (timeouts, versioning, auth, OAuth, retention, supply chain, warm-up, admission, minute cap) are each placed above;
- the setup items are scheduled (§1);
- **independent audit:** a Sonnet "AI hiring manager + staff engineer" pass found gaps, and all of them were folded in:
  - tiny GPT from scratch;
  - own sampler;
  - hybrid RAG;
  - VAD;
  - own CUDA kernels;
  - training systems;
  - DPO;
  - INT8 calibration;
  - production baseline rows;
  - gap rows;
  - the v7 split;
  - realistic estimates.
- HLD Phase-2 items (wake word, speaker verification, vision) stay future work, pointed to from the gaps register.

---

## 5. How each version runs

### Learn
Learn sessions are video first. Then Claude explains where the topic sits on your diagram, and you run a toy example.

**Resource spine.** Every link is verified with curl when `resources.md` is written.

| Version | Resources |
|---|---|
| F | Karpathy (*Intro*, *Deep Dive*, tokenizer, *Let's build GPT* (required)); 3Blue1Brown NN and transformers |
| v0, v1 | mlx-lm docs; OpenAI function-calling guide; Anthropic *Building effective agents* |
| v2 | SBERT; pgvector |
| v3, v4 | HF Audio Course ch.1, 5, 6 |
| v5 | Pipecat and LiveKit concepts |
| v6 | DeepLearning.AI *AI Agents in LangGraph* |
| v7a, v7b | Hamel Husain's eval posts; Google SRE book (SLO chapter) |
| v8 | ONNX Runtime docs; Triton Conceptual Guides 1–3 |
| v9 | vLLM PagedAttention blog; DeepLearning.AI *Efficiently Serving LLMs* |
| v10 | DeepLearning.AI *Quantization Fundamentals* and *Quantization in Depth* |
| v11 | DeepLearning.AI *Finetuning LLMs*; HF PEFT; mlx-lm LORA.md |
| v12 | GTC *How GPU Computing Works*; NVIDIA *Even Easier Intro to CUDA*; GPU MODE lectures |
| v13 | HF TRL and PEFT docs; AutoAWQ and GPTQ docs; vLLM LoRA docs |
| v14 | TensorRT quick start; Nsight guide |
| v15 | Locust or k6 docs; a coordinated-omission talk |
| v16 | Chip Huyen, *Designing ML Systems* (selected chapters); ML system-design interview guides |

### Build
Claude writes the code in small commits. For each step you **predict** first, then run, and Claude asks one "why" question. **You make one change yourself** each session.

**Learner-writes list (guards against passive learning).** For these core concept pieces, **you hand-type the code**. Claude gives you the signature and a failing test, then gives hints in steps (hint → partial → full), and reviews your code:

| Version | You write |
|---|---|
| F | Tiny GPT attention |
| v0 | Sampler |
| v1 | Tool-call loop |
| v6 | Approval state machine transitions |
| v8 | `config.pbtxt` |
| v10 | E3 quantiser |
| v11 | DPO loss |
| v12 | CUDA kernels |

### Close
- The demo.
- `EXPERIMENTS.md` entries.
- An HLD update if a decision changed.
- New rows in `production-gaps.md`.
- `docs/learn/vN.md`: your explain-it-back note.

### Interview session (I)
Claude runs a mock interview on that version's topics: 8–10 questions, one design question, and one "tell me about a time" story drawn from the build. You answer aloud or in writing **before** Claude reveals anything. Weak answers go into `LEARNING.md` and are re-asked after 1 week and again after 4 weeks (spaced repetition).

### GPU rule
Do the learning and the Mac dry run before renting. The paid sprint only executes a written checklist.

### Daily protocol (90 min; goes in `CLAUDE.md`, "Learning mode")
- **15 min orient and learn.** Claude places the task on the map in 3 sentences or fewer.
- **60 min build.**
- **15 min close.** A 3-question quiz, a `LEARNING.md` entry, a commit, and tomorrow's first step.
- **Claude rules:**
  - no unrequested scope;
  - when Claude cuts a production corner, it says so and logs it in the gaps register;
  - verify links;
  - prefer a great video when one exists.

## 6. v16: Interview capstone (6 sessions)
- **3 mock system-design rounds:**
  1. a voice assistant for 1M users;
  2. LLM serving at scale across multiple GPUs;
  3. a RAG plus agent platform.
- **A capacity-planning worksheet:** GPUs for N users, from tokens/s, KV memory and concurrency. Covers build vs buy and multi-tenant isolation.
- **1 deep-dive round** on your Fryday decisions.
- **1 ML-fundamentals rapid-fire round.**
- **Resume bullets and a portfolio README** built from the EXPERIMENTS numbers.

The output is `docs/learn/interview-kit.md`, which accumulates the per-version questions and STAR stories.
