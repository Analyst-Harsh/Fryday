# Fryday — High-Level Design

## Context
Draft source: `/Users/harshitgoyal/Dev/AI Projects/Fryday/docs/voice_assistant`.
Fryday is a Hinglish tap-to-talk voice assistant whose real purpose is **learning depth for AI engineering roles**: fine-tuning, audio basics, ONNX, CUDA working principles, TensorRT, Triton — inside a production-grade system. This session finalises the HLD and name; the implementation plan is the next session.

v2 incorporates three independent reviews (AI architect, production engineer, web fact-check). Main corrections: honest ASR latency, per-backend eval gate, GPU network/security, approval state machine, 16 GB Mac memory budget, realistic CI, verified component facts, sequenced scope.


## Decisions
| Topic | Decision |
|---|---|
| Name | **Fryday** |
| North star | Learning depth over product polish; target = AI engineering roles |
| Hardware | Apple M5, 16 GB, no CUDA → all NVIDIA work on rented GPU |
| GPU | RunPod L4 24 GB (~$0.39/h, verified Oct 2026), on demand in scripted sprints only |
| Runtime | Production-grade engineering, on-demand runtime; showcase = recorded demo + benchmark numbers |
| Serving | Same contracts on both machines. Triton for ASR/TTS/embeddings on GPU, and on Mac (CPU) **if spike 0 passes**, else plain ONNX Runtime/CT2 behind the same KServe v2 contract. LLM via OpenAI-compatible API: MLX-LM (native, not Docker) on Mac, vLLM on GPU |
| CUDA | Working principles through profiling; no kernel track |
| Vision | Phase 2 |
| Tools | Real: Swiggy MCP (Food/Instamart), Google Calendar, web search, notes, reminders |
| Spend confirmation | Read back stored order + spoken confirm phrase **and** UI tap |
| Tracing | Langfuse Cloud free tier, PII scrubbed before export |
| Router to outside model | Phase 2. Hosted models used offline only, as experiment baselines |

## 1. Components
1. **Web client** (thin React): tap-to-talk mic button (tap to start, tap to stop), WebSocket audio (browser echo cancellation on), ordered playback with jitter buffer, transcript, approval card with Confirm button.
2. **App service** (FastAPI, one deployable): gateway (WS, auth, admission control); turn manager (Silero VAD on CPU, barge-in); agent (LangGraph, approval interrupt); memory; text normaliser (script convention, numbers→words); MCP tool client; PII scrubber for outbound traces.
3. **Model layer**: Contract 1 KServe v2 gRPC → `asr`, `tts`, `embed`. Contract 2 OpenAI-compatible → LLM.
4. **Data**: Postgres + pgvector — users, memories, sessions, approvals, append-only audit log, history. Local filesystem volume for opt-in audio and model artefacts (object storage later). *Redis and MinIO cut for v1.*
5. **ML platform (offline)**: DVC for datasets; LoRA (PEFT on GPU, small MLX runs on Mac); eval harness; MLflow (local file store) as registry; PyTorch→ONNX→TensorRT export; release gate (§5).
6. **Observability**: OTel per-turn traces → Langfuse Cloud (scrubbed); Prometheus for app + Triton metrics. Grafana + DCGM only on GPU days.
7. **Infra**: docker compose with profiles (§6); GPU box via RunPod Terraform provider (early-stage) or `runpodctl` script; GitHub Actions CI.

## 2. Data flow (one turn)
1. Browser sends PCM16 16 kHz binary frames, each with `turn_id` + `seq`.
2. Second tap on the mic button ends the utterance (v1). Automatic end-of-speech detection (Silero VAD) is a later enhancement.
3. Utterance → `asr` → normaliser. Empty/low-confidence/no-speech output is dropped (Whisper hallucinates on silence).
4. Transcript → `embed` → pgvector top-k memories.
5. LangGraph → LLM (streaming). Tool call: Pydantic-validate → filler line → MCP. Invalid call: one repair attempt using the validation error, then polite refusal.
6. Reply tokens → chunks (short first chunk ~6 words, then sentences) → normalise → `tts` → audio emitted **in order** with `turn_id`/`chunk_seq`, bounded queue (backpressure pauses TTS).
7. Background: memory extraction, audit/history rows, audio saved only with consent.

**Barge-in**: tapping the mic during playback stops the reply and starts listening; it bumps the turn epoch → client drops stale chunks; LLM stream closed (vLLM aborts and frees KV blocks); in-flight Triton gRPC calls cancelled; turn marked interrupted. Tap-to-talk avoids server-side VAD on echoing audio in v1.

**Latency budget** — hypotheses to measure, not promises:
| Stage | GPU (L4) | Mac |
|---|---|---|
| End of utterance | ~0 (second tap) / 200 ms (VAD, later) | same |
| ASR (scales with length, RTF-based; 3–5 s utterance) | 300–500 ms | 1–2 s (spike) |
| Memory | 50 ms | 80 ms |
| LLM first token | 250 ms | ~600 ms |
| TTS first audio | 200–400 ms | spike |
| Mac↔GPU tunnel RTT | +30–150 ms | — |
| **First audio** | **~1.1–1.4 s** | **~3 s, dev only** |
Stretch to approach ~850 ms: streaming-native ASR candidate and/or chunked pseudo-streaming with early LLM start.

Admission control: N live conversations per backend, N set by load test; beyond that, "busy".

## 3. Models
| Role | Candidates | Work | Mac build | GPU build |
|---|---|---|---|---|
| ASR | Whisper large-v3-turbo; open Hinglish Whisper; one streaming-native model (Conformer/Parakeet family) | LoRA for named entities; quantise | CT2 int8 or ONNX (Python/ORT backend) | CT2 fp16/int8 via Triton **Python backend** (no official CT2 backend); TensorRT-LLM Whisper encoder as experiment |
| LLM | 3–4B open models | LoRA SFT: tool calls + spoken Hinglish; merge; 4-bit | MLX 4-bit (tool parsing model-dependent; no server-side structured output → validate-and-repair) | vLLM AWQ/GPTQ (verify on L4) |
| TTS | Indic Parler-TTS (Apache 2.0, gated), Magpie multilingual 357M (NVIDIA Open Model License) — both support Hindi | Pick, fix one voice; no fine-tune in v1 | PyTorch/NeMo in Triton Python backend | same; ONNX/TRT = stretch (unproven) |
| Embeddings | bge-m3 (568M, MIT; fallback multilingual-e5-small on Mac) | none — chosen as the easy TensorRT target | ONNX | TensorRT fp16 |
24 GB estimate (spike, not fact): ASR ~2 + LLM ~3 + 6 KV cap + TTS ~2–3 + embed ~1.1 + CUDA contexts/Triton overhead ≈ 16–18 GB.

## 4. Learning map
| Topic | Where | Concrete deliverable |
|---|---|---|
| Audio basics | resampling 48k→16k, log-mel, VAD, codecs, how TTS vocodes | notes + tested feature module (time-boxed) |
| Fine-tuning | LLM LoRA first, then ASR LoRA; synthetic data gen + filtering | before/after: tool accuracy, WER, entity accuracy; constrained decoding vs fine-tune vs both |
| Quantisation | from-scratch int8/int4 weight quantiser on one layer, then AWQ vs GPTQ vs MLX | error analysis + quality/latency table per format |
| LLM serving | KV-cache size formula vs vLLM reality; paged attention; continuous batching | TTFT/throughput vs concurrency curve |
| ONNX | export embed, ASR encoder | parity tests vs PyTorch (tolerances), Mac latency |
| TensorRT | embed (easy) → Whisper encoder via TRT-LLM (hard); TRT-LLM for the LLM = stretch | fp32/fp16 engines, dynamic shapes; PyTorch vs ONNX vs TRT table |
| CUDA principles | Nsight Systems on GPU serving: streams, H↔D copies, pinned memory, launch overhead | one annotated request timeline + CUDA graphs on/off delta. Note: GPU counters may need CAP_SYS_ADMIN, check provider before sprint 2 |
| Triton | model repo, config.pbtxt, dynamic batching, instance groups, ensemble, metrics, Model Analyzer | one Triton vs separate servers; vLLM backend vs standalone vLLM; load test |
| SWE | CI, contract tests, tracing, idempotency, IaC, release gate | working pipeline + EXPERIMENTS.md (negative results included) + cost per conversation-minute |

## 5. Evaluation and release gate
- **Frozen sets**: ASR = real voices only, speakers absent from training; LLM = 200 hand-reviewed tool conversations; plus a synthetic-only ASR dev set to expose TTS-quirk overfitting.
- **Hygiene**: dedup/template-overlap check train vs test; frozen sets never used for tuning; usage counter on each frozen set.
- **Judge**: different model from the data generator; calibrated on a human-labelled subset with a minimum agreement threshold.
- **Gate is a matrix**: one row per (model, backend, quant format) with its own thresholds; report confidence intervals; promotion margin must exceed noise. Mac vs GPU divergence metric = tool-call agreement on the same frozen set.
- **Baselines**: hosted ASR/LLM run offline on the frozen sets as the reference.
- Promote in MLflow only the exact artefact that will be served.

## 6. Running it: environments, network, cost
**Mac (16 GB) compose profiles**: `core` = app + Postgres + model backends (MLX-LM native on host); `obs` = Prometheus (opt-in). Langfuse is cloud, so no local tracing stack. Memory checked in spike 0.
**GPU box**: Triton + vLLM only — inference, no user data, no OAuth tokens. Reached **only through Tailscale/WireGuard**; no public ports; services bound to tunnel interface; vLLM `--api-key`. Secrets injected at boot from local env; disk wiped on teardown. GPU region chosen for lowest RTT.
**Cost guards**: auto-teardown timer + dead-man switch (box self-terminates without heartbeat) + provider budget alert + sprint checklist. Per-user daily minute cap.
**GPU sprints** (box destroyed after each): (1) LLM + ASR fine-tuning, (2) TensorRT + GPU Triton + profiling, (3) benchmarks + load test.

## 7. Safety and security
- **Approval state machine** in Postgres: `pending → approved → executing → done | failed | expired | unknown`, transitions by compare-and-set (`UPDATE … WHERE status='pending'`), expiry by DB clock (~45 s).
- Approval bound to a **hash of the exact action payload** (items, qty, address, price); read-back text generated from the stored row, never by the LLM — so injected text cannot shape an approval.
- **Spend actions**: confirm phrase + UI tap; low-confidence ASR rejected. Idempotency key persisted before the MCP call. **Spend actions are never auto-retried**: on timeout, state = `unknown`, then query order status. Swiggy orders pass through `PENDING_PAYMENT` with UPI via its MCP payment tools; access and limits via Swiggy Builders programme (to obtain).
- Non-spend tools: one retry with the same idempotency key.
- **Audit log** append-only (INSERT-only DB role).
- Tool output and web pages treated as data; injection suite in CI.
- **Auth**: single-user; session cookie + Origin check on WS; WS token sent in first message, never in URL. OAuth tokens encrypted at rest (Fernet, key from env), least scopes (calendar events only).
- **Privacy**: audio off by default, opt-in, deletable. Transcripts scrubbed (phone, address, names) before export to Langfuse Cloud.
- **Known v1 risk**: no speaker verification; the UI tap is the mitigation.

## 8. Failure handling
| Failure | Behaviour |
|---|---|
| Backend cold / box gone / tunnel drop | readiness fails → "warming up / GPU offline"; config can switch to Mac backend; in-flight turn fails cleanly |
| Per-stage stall | per-stage timeouts (ASR, LLM first token, TTS) → apologise, end turn |
| Tool error (non-spend) | one retry, same key, then tell the user |
| Spend timeout | `unknown` state, status lookup, never re-send |
| OAuth refresh failure / MCP schema change / rate limit | tool disabled with a clear message; contract test catches schema drift |
| ASR hallucination on silence/noise | no-speech/low-confidence filter drops it |
| WS drop | conversation history kept; in-flight turn restarted, not resumed; pending approvals expire |
| Duplicate WS for same user | newest wins, old closed |
| Postgres down | service unhealthy, refuse turns |
| Overload | admission control → busy |

## 9. Testing and CI
- **CI (GitHub Actions)**: unit tests (normaliser, chunker, approval state machine, memory), contract tests against a **stub** KServe/OpenAI server, MCP tests with recorded responses, injection suite. CI does not run real models — stated explicitly.
- **Local `make e2e`**: real models, recorded Hinglish audio replayed end-to-end; behavioural contract checks (tool-call JSON schema, chat-template parity) parametrised by backend URL.
- **GPU sprint script**: runs the same e2e + contract suite against the GPU backend, plus k6/Locust WS load test; results committed to EXPERIMENTS.md.
- Live Swiggy/Google calls: manual smoke test only.

## 10. Build sequence (milestones)
0. **Spikes (hard gate before the implementation plan is finalised)**: Triton arm64 CPU in Docker on M5; Mac memory with `core` profile; MLX tool calling with candidate LLMs; Swiggy Builders access.
1. Text-only agent on Mac: real tools, approvals, memory, LLM eval harness.
2. Voice on Mac: ASR + TTS + WS client, end to end.
3. LLM LoRA fine-tune (sprint 1) + quantisation study.
4. Triton + ONNX→TensorRT on embeddings; GPU Triton for all models; profiling (sprint 2).
5. ASR LoRA + TRT-LLM Whisper encoder experiment.
6. Benchmarks, load test, cost per minute, demo recording (sprint 3).

## Phase 2 (not v1)
Vision (image Q&A, ViT/SigLIP→ONNX→TRT), router to an outside model, phone app + wake word, speaker verification, end-of-turn model, proactive reminders, smart-home tools, custom CUDA kernel.

## Verified facts (fact-check, Oct 2026)
Silero VAD MIT, <1 ms/chunk CPU. Swiggy MCP has Food/Instamart/Dineout with UPI payment flow. Langfuse v3 self-host = Postgres + ClickHouse + Redis + S3 (why we use cloud). Triton vLLM backend exists. TRT-LLM Whisper uses separate encoder/decoder engines. RunPod has official (early) Terraform provider; Lambda has no L4.
**Still unverified** → spike 0: Triton on Apple Silicon Docker, MLX tool calling per model, TTS ONNX export, Nsight counters on RunPod, 24 GB fit.
