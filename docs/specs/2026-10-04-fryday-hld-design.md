# Fryday — High-Level Design

## Context
Draft source: `/Users/harshitgoyal/Dev/AI Projects/Fryday/docs/voice_assistant`.
Fryday is a Hinglish tap-to-talk voice assistant whose real purpose is **learning depth for AI engineering roles**: fine-tuning, audio basics, ONNX, CUDA working principles, TensorRT, Triton — inside a production-grade system. The implementation plan is a separate document.

**Revision history**
- **v2** applied three independent reviews: an AI architect, a production engineer and a web fact-check. The main changes were:
  - an honest ASR latency budget;
  - a per-backend evaluation gate;
  - network and security for the rented GPU;
  - the approval state machine;
  - a memory budget for the 16 GB Mac;
  - a realistic CI scope;
  - verified component facts;
  - a sequenced scope.
- **v3** applied a second fact-check and a production-readiness review. The main changes were:
  - money-path crash recovery and at-most-once semantics;
  - an operations section covering SLOs, backups, a runbook, supply chain and provenance;
  - a per-hop latency budget with warm-up;
  - Tailscale userspace networking on RunPod;
  - exact model IDs;
  - script convention and Hindi text normalisation moved to spike 0;
  - tap-to-talk input, with VAD removed from v1.

## Decisions
| Topic | Decision |
|---|---|
| Name | **Fryday** |
| North star | Learning depth over product polish; target = AI engineering roles |
| Hardware | Apple M5, 16 GB, no CUDA → all NVIDIA work on rented GPU |
| GPU | RunPod L4 24 GB, from ~$0.39/h on Community Cloud (Secure Cloud costs more; re-check at booking), on demand in scripted sprints only |
| Runtime | Production-grade engineering, on-demand runtime; showcase = recorded demo + benchmark numbers |
| Serving | Same contracts on both machines. Triton for ASR, TTS and embeddings on the GPU, and on the Mac (CPU) **if spike 0 passes**; otherwise plain ONNX Runtime/CT2 behind the same KServe v2 contract. LLM through an OpenAI-compatible API: MLX-LM (native, not Docker) on the Mac, vLLM on the GPU |
| Voice input | Tap to start, tap to stop (v1). Automatic end-of-speech detection is a later enhancement |
| Script convention | **Decided in spike 0**. Option A: Roman Hinglish inside the pipeline, converted to Devanagari only for TTS. Option B: Devanagari for Hindi words everywhere |
| CUDA | Working principles through profiling; no kernel track |
| Vision | Phase 2 |
| Tools | Real: Swiggy MCP (Food/Instamart), Google Calendar, web search, notes, reminders |
| Spend confirmation | Read back the stored order + amount shown as text in the UI + spoken confirm phrase **and** UI tap, both bound to the same approval |
| Spend delivery | At-most-once with reconciliation (exactly-once only if spike 0 shows Swiggy honours idempotency keys) |
| Tracing | Langfuse Cloud Hobby tier (50k units/month, 30-day retention; region chosen at signup and fixed), PII scrubbed before export |
| Router to outside model | Phase 2. Hosted models used offline only, as experiment baselines |

## 1. Components
1. **Web client** (thin React):
   - Tap-to-talk mic button: one tap starts listening, a second tap stops.
   - WebSocket audio, with browser echo cancellation on.
   - Ordered playback through a jitter buffer.
   - Transcript.
   - Approval card showing the amount, items and address **as text**, with a Confirm button.
2. **App service** (FastAPI, one deployable):
   - gateway: WebSocket, auth, admission control, protocol version field;
   - turn manager: tap boundaries, barge-in epochs, cancellation;
   - agent: LangGraph with a checkpointer and an approval interrupt;
   - memory;
   - text normaliser (§3.1);
   - MCP tool client;
   - spend reconciler (§7);
   - PII scrubber for outbound traces **and** local logs.
3. **Model layer**:
   - Contract 1: KServe v2 gRPC → `asr`, `tts`, `embed`.
   - Contract 2: OpenAI-compatible → LLM.
4. **Data**:
   - Postgres + pgvector holds users, memories, sessions, approvals, the append-only audit log and history. Schema changes go through Alembic migrations.
   - A local filesystem volume holds opt-in audio and model artefacts; object storage comes later.
   - *Redis and MinIO are cut for v1.*
5. **ML platform** (offline):
   - DVC for datasets.
   - LoRA: PEFT on the GPU, small MLX runs on the Mac.
   - Eval harness.
   - MLflow (local file store) for runs and the registry.
   - **`models.lock`** (git-tracked) pins exactly what is served (§5).
   - PyTorch→ONNX→TensorRT export.
   - Release gate (§5).
6. **Observability**:
   - OTel per-turn traces → Langfuse Cloud (scrubbed).
   - Prometheus for app and Triton metrics, plus local SLO alerts (§8).
   - Grafana and DCGM only on GPU days.
7. **Infra**:
   - docker compose with profiles (§6).
   - GPU box via `runpodctl` script; the RunPod Terraform provider is optional and early-stage.
   - GitHub Actions CI.

## 2. Data flow (one turn)
1. Tap to start: the browser streams PCM16 16 kHz binary frames, each tagged with `turn_id` and `seq`.
2. Tap to stop ends the utterance.
3. Utterance → `asr` → normaliser (in). Empty, low-confidence or no-speech output is dropped, because Whisper hallucinates on silence.
4. Transcript → `embed` → pgvector top-k memories.
5. LangGraph → LLM (streaming).
   - **Tool call:** Pydantic-validate → filler line → MCP.
   - **Invalid call:** one repair attempt using the validation error, then a polite refusal.
6. Reply tokens → chunks (a short first chunk of ~6 words, then sentences) → normaliser (out) → `tts` → audio emitted **in order** with `turn_id`/`chunk_seq`. The queue is bounded, so backpressure pauses TTS.
7. Background: memory extraction, audit and history rows; audio is saved only with consent.

**Barge-in**: tapping the mic during playback stops the reply and starts listening. It does five things:
1. bumps the turn epoch, so the client drops stale chunks;
2. closes the LLM stream (vLLM aborts the request);
3. cancels in-flight Triton requests (**cancellation is gRPC-only in Triton; the Python and vLLM backends support early termination**);
4. marks the turn interrupted.

Tap-to-talk means no VAD runs on echoing audio in v1.

**Warm-up**: each backend runs one warm-up inference per model before its readiness probe goes green, because Triton cold start and the first TensorRT/CUDA-graph call take seconds.

**Latency budget**: hypotheses to measure, not promises. On the GPU path, every call from the Mac to the GPU pays the tunnel round trip (RTT).

| Stage | GPU (L4) | Mac |
|---|---|---|
| End of utterance | ~0 (second tap) | same |
| ASR (scales with length, RTF-based; 3–5 s utterance) | 300–500 ms + 1 RTT | 1–2 s (spike) |
| Memory (embed call + pgvector) | 50 ms + 1 RTT | 80 ms |
| LLM until the first ~6-word chunk exists | 250 ms + ~10 tokens + 1 RTT | ~600 ms + tokens |
| TTS first audio | 200–400 ms + 1 RTT | spike |
| Tunnel RTT (×4 hops) | 30–150 ms each | — |
| **First audio** | **~1.4–1.8 s** | **~3 s, dev only** |

Stretch goal, approaching ~1 s:
- choose a GPU region close to you;
- co-locate the app with the GPU during benchmarks;
- try a streaming-native ASR;
- use chunked pseudo-streaming with an early LLM start.

Admission control: N live conversations per backend, with N set by the load test; beyond that, the user hears "busy".

## 3. Models
Exact IDs were checked against model cards on 2026-10-04 unless marked *(verify)*. Every served model is pinned in `models.lock` (§5).

| Role | Candidates | Work | Mac build | GPU build |
|---|---|---|---|---|
| ASR | `openai/whisper-large-v3-turbo` *(licence verify)*; `Oriserve/Whisper-Hindi2Hinglish-Apex` (Apache-2.0, outputs Roman Hinglish), `…-Prime` as the alternative; streaming-native: NVIDIA Nemotron ASR with Hindi or AI4Bharat IndicConformer *(both verify ID, Hindi, streaming)* | LoRA for named entities; quantise | CT2 int8 or ONNX (Python/ORT backend) | CT2 fp16/int8 via the Triton **Python backend** (no official CT2 backend); TensorRT-LLM Whisper encoder as an experiment |
| LLM | `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0); `meta-llama/Llama-3.2-3B-Instruct` (Llama 3.2 licence, gated, Hindi officially supported); `google/gemma-3-4b-it` (Gemma licence, gated; no native function calling); check for newer 2026 3–4B releases | LoRA SFT for tool calls and spoken Hinglish; merge; 4-bit | MLX 4-bit via `mlx_lm.server` (OpenAI-compatible, streaming; tool-call parsing per model and streaming+tools **unverified** → spike S0-3; validate-and-repair) | vLLM AWQ/GPTQ (verify on L4) |
| TTS | `ai4bharat/indic-parler-tts` (Apache-2.0, gated); `nvidia/magpie_tts_multilingual_357m` (NVIDIA Open Model License; batched/sliding-window synthesis, **not** native streaming). Both support Hindi | Pick one and fix one voice; no fine-tuning in v1 | PyTorch/NeMo in the Triton Python backend | Same, as chunked unary calls; ONNX/TRT is a stretch (unproven) |
| Embeddings | `BAAI/bge-m3` (MIT); fallback `intfloat/multilingual-e5-small` on the Mac *(licence verify)* | None; chosen as the easy TensorRT target | ONNX | TensorRT fp16 |
| VAD (later) | Silero VAD (MIT) | — | CPU | — |

24 GB estimate (a spike, not a fact): ASR ~2 + LLM ~3 + 6 KV cap + TTS ~2–3 + embed ~1.1 + CUDA contexts/Triton overhead ≈ 16–18 GB.

### 3.1 Text normaliser
- **Out (before TTS): rules, no model.**
  - Rules cover currency, time, date, plain numbers and abbreviations, turning them into Hindi words.
  - NVIDIA NeMo text processing supports Hindi only for *inverse* normalisation (words → numbers). Its Hindi numbers→words support is unverified, so plan a small custom rule set and adopt NeMo only if spike 0 shows it works.
  - Golden unit tests cover every rupee amount and number format.
- **In (after ASR): the script convention, decided in spike 0.**
  - Option A: keep Roman Hinglish, which Hinglish ASR models output natively, and transliterate to Devanagari only before TTS.
  - Option B: fine-tune the ASR to output Devanagari for Hindi words.
  - Either way, transliteration (e.g. AI4Bharat IndicXlit *(verify)*) is the fallback.
- **Money safety does not depend on TTS:** amounts are always shown as text on the approval card.

## 4. Learning map
| Topic | Where | Concrete deliverable |
|---|---|---|
| Audio basics | resampling 48k→16k, log-mel, VAD, codecs, how TTS vocodes | notes + tested feature module (time-boxed) |
| Fine-tuning | LLM LoRA first, then ASR LoRA; synthetic data gen + filtering | before/after: tool accuracy, WER, entity accuracy; constrained decoding vs fine-tune vs both |
| Quantisation | from-scratch int8/int4 weight quantiser on one layer, then AWQ vs GPTQ vs MLX | error analysis + quality/latency table per format |
| LLM serving | KV-cache size formula vs vLLM reality; paged attention; continuous batching | TTFT/throughput vs concurrency curve |
| ONNX | export embed, ASR encoder | parity tests vs PyTorch (tolerances), Mac latency |
| TensorRT | embed (easy) → Whisper encoder via TRT-LLM (hard); TRT-LLM for the LLM = stretch | fp32/fp16 engines, dynamic shapes; PyTorch vs ONNX vs TRT table |
| CUDA principles | Nsight Systems on GPU serving: streams, H↔D copies, pinned memory, launch overhead | One annotated request timeline + CUDA graphs on/off delta. GPU counters need `--cap-add=SYS_ADMIN`, and a running DCGM can block them; check before sprint 2 |
| Triton | model repo, config.pbtxt, dynamic batching, instance groups, ensemble, metrics; Model Analyzer = stretch | one Triton vs separate servers; vLLM backend vs standalone vLLM; load test |
| SWE | CI, contract tests, tracing, idempotency, IaC, release gate, operations (§8) | working pipeline + EXPERIMENTS.md (negative results included) + cost per conversation-minute |

## 5. Evaluation, release gate and model provenance
- **Frozen sets:**
  - ASR: real voices only, from speakers absent from training.
  - LLM: 200 hand-reviewed tool conversations.
  - A synthetic-only ASR dev set, to expose overfitting to TTS quirks.
- **Hygiene:** dedup and template-overlap checks between train and test; frozen sets never used for tuning; a usage counter on each frozen set.
- **Judge:** a different model from the data generator, calibrated on a human-labelled subset with a minimum agreement threshold.
- **The gate is a matrix:**
  - one row per (model, backend, quant format), each with its own thresholds;
  - confidence intervals reported, and the promotion margin must exceed the noise;
  - a Mac-vs-GPU divergence metric: tool-call agreement on the same frozen set.
- **Baselines:** hosted ASR/LLM run offline on the frozen sets as the reference.
- **Provenance:** every served model has a `models.lock` entry with:
  - its HF repo and revision SHA;
  - its file SHA-256;
  - its licence;
  - the training-data DVC hash;
  - the git commit;
  - the gate result.
  
  Only the exact artefact that passed the gate is served.
- **Reproducibility:** training runs record seeds, library versions and the base-model revision.
- **Rollback:** repoint `models.lock` to the previous promoted artefact and restart the backend.

## 6. Running it: environments, network, cost
**Mac (16 GB) compose profiles:**
- `core`: app + Postgres + model backends, with MLX-LM native on the host.
- `obs`: Prometheus (opt-in).
- Langfuse is in the cloud, so there is no local tracing stack.
- Memory is checked in spike 0.

**GPU box:**
- Runs Triton + vLLM only, as inference: it **stores** no user data and no OAuth tokens, though it sees live audio and text in transit.
- Reached **only over Tailscale**; no public ports; vLLM `--api-key`.
- RunPod pods usually have no TUN device, so `tailscaled` runs in **userspace-networking** mode. Services bind to localhost and are exposed to the tailnet with `tailscale serve`, or reached through the tailscaled proxy. The sprint-2 preflight checks this.
- Secrets are injected at boot from a local env file, and the disk is wiped on teardown.
- The GPU region is chosen for the lowest RTT.

**Cost guards:** an auto-teardown timer, a dead-man switch (the box self-terminates without a heartbeat), a provider budget alert, a sprint checklist, and a per-user daily minute cap.

**GPU sprints** (the box is destroyed after each):
1. LLM + ASR fine-tuning.
2. TensorRT + GPU Triton + profiling.
3. Benchmarks + load test.

## 7. Safety and security
- **Approval state machine** in Postgres: `pending → approved → executing → done | failed | expired | unknown`.
  - `pending` expires after ~45 s and `approved` after ~10 s, both by the **DB clock**.
  - Approve and expire are both compare-and-set: `UPDATE … WHERE status='pending' AND expires_at > now()`.
- Each approval is bound to a **hash of the exact action payload** (items, qty, address, price). The read-back text is generated from the stored row, never by the LLM, so injected text cannot shape an approval.
- **Spend confirmation:**
  - The confirm phrase **and** the UI tap must both carry the same `approval_id` + `payload_hash`.
  - Low-confidence ASR is rejected.
  - The approval card shows the amount as text.
- **Spend delivery is at-most-once with reconciliation:**
  - The idempotency key is persisted before the MCP call.
  - Spend actions are **never auto-retried**. On timeout the state becomes `unknown`, then the order status is queried.
  - A **reconciler** runs at startup and every 60 s. It moves any `executing` row older than 30 s to `unknown` and resolves each `unknown` through an order-status lookup.
  - For Swiggy, `done` means `confirm_order` returned `PLACED`. `PENDING_PAYMENT` that ends `FAILED` maps to `failed`. Payment runs through the MCP tools `get_payment_options`, `check_payment_status` and `confirm_order`. Access is through the Swiggy **Builders Club** (approval needed).
- **Non-spend tools:** one retry with the same idempotency key.
- **Audit log:** append-only. The app's DB role can only INSERT into it; migrations use a separate role.
- **Untrusted input:** tool output and web pages are treated as data, never instructions. An injection test suite runs in CI.
- **Auth:**
  - Single user, with one static access token.
  - The app listens only on localhost or the tailnet, and checks the Origin header on the WebSocket.
  - The token is sent in the first WebSocket message, never in the URL.
- **OAuth tokens:**
  - Encrypted at rest with Fernet; the key comes from env and is stored apart from backups.
  - Least scopes: Google Calendar `calendar.events.owned`.
- **Privacy and retention:**
  - Audio is off by default, opt-in, and deletable.
  - Transcripts and turns are kept 30 days.
  - A `forget` command deletes the user's memories, history and audio.
  - The PII scrubber (phone, address, names) applies to the Langfuse export **and** local logs.
- **Supply chain:**
  - Container images pinned by digest.
  - Dependency lockfile; `pip-audit` and Trivy in CI.
  - `gitleaks` pre-commit hook.
  - Model weights pinned by revision + hash (§5).
- **Known v1 risk:** there is no speaker verification; the UI tap is the mitigation.

## 8. Operations
**SLOs** (initial targets, tuned after the first measurements):

| SLI | SLO |
|---|---|
| p95 time to first audio | ≤ 2.0 s on GPU, ≤ 4 s on Mac |
| Turn success rate (no timeout or error) | ≥ 98% |
| Unresolved spend `unknown` | 0 after 5 min |

**Alerts:** a small script checks Prometheus and raises a desktop notification on an SLO breach, a stuck `unknown` or a missed GPU heartbeat.

**Timeouts** (initial values):

| Stage | Timeout |
|---|---|
| ASR | 3 s |
| LLM first token | 2 s |
| TTS chunk | 2 s |
| MCP non-spend | 5 s |
| MCP spend | 15 s, then `unknown` |

**Backups:** a nightly `pg_dump` to a local encrypted folder, keeping 7. A restore drill runs quarterly. The Fernet key is kept separate from the dumps.

**Runbook:** `RUNBOOK.md` covers:
- GPU box dead or tunnel down;
- a stuck `unknown` approval;
- an expired OAuth token;
- Postgres restore;
- model rollback.

**Versioning:**
- The WebSocket protocol carries a `v` field.
- Model contracts are versioned by model name and version in the Triton repository.
- DB changes go through Alembic migrations, run on startup.

## 9. Failure handling
| Failure | Behaviour |
|---|---|
| Backend cold / box gone / tunnel drop | Readiness fails (warm-up not done) → "warming up / GPU offline"; config can switch to the Mac backend (manual, by design); the in-flight turn fails cleanly |
| Per-stage stall | Timeouts from §8 → apologise and end the turn |
| Tool error (non-spend) | One retry with the same key, then tell the user |
| Spend timeout | `unknown` state, status lookup, never re-send |
| Crash during a spend | Reconciler moves stale `executing` → `unknown` → resolved by status lookup |
| OAuth refresh failure / MCP schema change / rate limit | Tool disabled with a clear message; the contract test catches schema drift |
| ASR hallucination on silence or noise | The no-speech/low-confidence filter drops it |
| WS drop | Conversation history kept; the in-flight turn is restarted, not resumed; pending approvals expire |
| Duplicate WS for the same user | Newest wins; the old one is closed |
| Postgres down | Service unhealthy, refuses turns; restore from backup per the runbook |
| Overload | Admission control → busy |

## 10. Testing and CI
- **CI (GitHub Actions):**
  - Unit tests: normaliser golden tests for amounts, numbers and dates; chunker; approval state machine including the races and the reconciler; memory.
  - Contract tests against a **stub** KServe/OpenAI server.
  - MCP tests with recorded responses.
  - Injection suite.
  - Migration test.
  - `pip-audit`, Trivy and gitleaks.
  - CI does not run real models, and this is stated explicitly.
- **Local `make e2e`:**
  - Real models, with golden Hinglish audio fixtures replayed end to end.
  - Behavioural contract checks (tool-call JSON schema, chat-template parity), parametrised by backend URL.
  - Chaos-lite: kill a backend mid-turn and the GPU tunnel mid-spend, and check recovery.
- **GPU sprint script:** runs the same e2e and contract suite against the GPU backend, plus a k6/Locust WebSocket load test. Results are committed to EXPERIMENTS.md.
- **Manual:** live Swiggy and Google calls are a smoke test only.

## 11. Build sequence (milestones)
0. **Spikes**: a hard gate before the implementation plan is finalised.
   - Triton arm64 CPU in Docker on the M5.
   - Mac memory with the `core` profile.
   - MLX tool calling and streaming with the candidate LLMs.
   - Swiggy Builders Club access, limits, and whether idempotency keys are honoured.
   - Script convention, A vs B, on real samples.
   - Hindi numbers→words: NeMo vs custom rules.
   - Exact IDs and licences for the *(verify)* models.
   - Tailscale userspace mode on a RunPod pod.
1. **Text-only agent on the Mac:** real tools, approvals plus reconciler, memory, LLM eval harness.
2. **Voice on the Mac:** ASR + TTS + WebSocket client, end to end.
3. **LLM LoRA fine-tune** (sprint 1) + quantisation study.
4. **GPU serving:** Triton + ONNX→TensorRT on embeddings; GPU Triton for all models; profiling (sprint 2).
5. **ASR:** ASR LoRA, trained in sprint 1 after the LLM, and the TRT-LLM Whisper encoder experiment in sprint 2.
6. **Prove it:** benchmarks, load test, cost per minute, demo recording (sprint 3).

## Phase 2 (not v1)
- Vision (image Q&A, ViT/SigLIP→ONNX→TRT).
- Automatic end-of-speech detection (Silero VAD + end-of-turn model).
- Router to an outside model.
- Phone app + wake word.
- Speaker verification.
- Proactive reminders.
- Smart-home tools.
- Custom CUDA kernel.

## Verified facts (fact-checks, 2026-10-04)
- **Swiggy MCP:** has Food, Instamart and Dineout, with a UPI flow (`PENDING_PAYMENT` → `confirm_order` → `PLACED`/`FAILED`) and Builders Club approval.
- **Triton:**
  - Cancellation is gRPC-only.
  - Decoupled (streaming) models need gRPC streaming.
  - The vLLM backend exists.
- **TRT-LLM Whisper** uses separate encoder and decoder engines.
- **LangGraph `interrupt`** needs a checkpointer and a `thread_id`.
- **Langfuse:**
  - Self-hosting v3 needs Postgres, ClickHouse, Redis and S3, which is why we use the cloud.
  - The Hobby tier is 50k units per month with 30-day retention.
- **RunPod and Lambda:**
  - RunPod has an official Terraform provider, still early.
  - Lambda lists no L4.
  - RunPod pods typically need Tailscale userspace networking.
- **Nsight counters** need SYS_ADMIN.
- **Model licences:** Qwen3-4B-Instruct-2507 Apache-2.0; Llama-3.2-3B gated with Hindi supported; Indic Parler-TTS Apache-2.0 gated; Magpie NVIDIA Open Model License; bge-m3 MIT; Oriserve Hinglish Whisper Apache-2.0.

**Still unverified → spike 0:**
- Triton on Apple Silicon Docker.
- MLX streaming plus tool calling.
- Streaming ASR with Hindi.
- Hindi numbers→words in NeMo.
- IndicXlit.
- Whisper turbo and e5-small licences.
- vLLM AWQ/GPTQ on L4.
- TTS ONNX export.
- DCGM on RunPod.
- 24 GB fit.
