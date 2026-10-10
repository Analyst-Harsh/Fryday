# Production gaps register

This register lists every place where Fryday deliberately does less than a production team would. Each row says:
- what production teams do;
- what we do instead;
- why we skip it now, and when it would start to matter;
- the interview talking point.

Being able to explain these trade-offs is the point of the register. Add a row whenever a version cuts a corner (roadmap §2), and note the version that added it.

| # | Production standard | Fryday's choice | Why not now / when it matters | Interview talking point | Added |
|---|---|---|---|---|---|
| 1 | WebRTC (UDP, Opus, TURN/STUN) for real-time voice | WebSocket with PCM16 frames | One user on localhost. It matters on mobile or lossy networks, where TCP head-of-line blocking adds jitter. | Why voice products use WebRTC: UDP, packet-loss concealment, built-in AEC, NAT traversal. Why WebSocket is fine for a prototype. | plan |
| 2 | Kubernetes, autoscaling on GPU metrics, sticky WebSocket load balancing | docker compose plus scripts | Single user, one box. It matters with more than one replica or real traffic. | HPA on custom metrics such as queue depth or KV-cache usage; why WebSockets need sticky sessions or a session store. | plan |
| 3 | Terraform / IaC | Provider-CLI `gpu-up`/`gpu-down` scripts | One provider and boxes that only live for a few hours. It matters with several environments or a team. | Declarative vs imperative infrastructure, state files, drift. | plan |
| 4 | Secrets manager (Vault, AWS SSM) plus KMS | `.env` file plus Fernet key from env | Local, single user. It matters on shared infrastructure, for rotation, and for auditing. | Secret rotation, envelope encryption, least privilege. | plan |
| 5 | Multi-tenant auth, OAuth login, per-user rate limits | Static token, localhost only, global daily minute cap (v5) | Single user. It matters as soon as there is a second user. | Token-bucket rate limiting; tenant isolation for data and compute. | plan |
| 6 | Canary / blue-green model rollout, shadow traffic | Offline eval gate plus `models.lock` rollback | There is no traffic to split. It matters with real users. | Offline gates vs online canaries; what to monitor during a rollout. | plan |
| 7 | Model registry service, feature store | MLflow local file store plus git | Solo scale. It matters with a team, many models, or online features. | Registry stages, lineage, train/serve skew. | plan |
| 8 | Managed Postgres with point-in-time recovery and encryption at rest | Local Postgres plus a nightly encrypted dump | Learning scale. It matters for real user data. | RPO/RTO, WAL archiving vs dumps. | plan |
| 9 | SBOM plus image signing (cosign), SLSA provenance | Digest pinning, pip-audit, Trivy | We start with the cheap subset. It matters when publishing images or working under compliance rules. | Supply-chain attacks; what SLSA levels guarantee. | plan |
| 10 | On-call paging (PagerDuty) with escalation | A desktop notification | Solo. It matters once anyone depends on the service. | Alert on symptoms, not causes; error-budget burn alerts. | plan |
| 11 | Multi-GPU tensor and pipeline parallelism | One 24 GB GPU | A 4B model fits on one card. It matters for 70B+ models or for lower latency. | Tensor parallelism (all-reduce per layer) vs pipeline parallelism (bubbles); when to use each. | plan |
| 12 | RLHF with PPO or GRPO and a reward model | SFT LoRA plus a small DPO run (v11) | Cost, and no preference data at scale. | PPO vs GRPO vs DPO; reward hacking. | plan |
| 13 | Speaker verification, wake word, learned end-of-turn model | Tap-to-talk plus a UI tap; Silero VAD as an option (v5) | Scope (HLD Phase 2). It matters for hands-free use or for spend safety. | Endpointing trade-off: cutting the user off vs added latency. | plan |
| 14 | DPDP/GDPR process, data processing agreements with vendors | Retention, `forget`, PII scrubbing | Personal project. It matters once there are any external users. | Data minimisation; the right to erasure across backups. | plan |
| 15 | Streaming ASR with partial transcripts | Batch ASR on the tapped utterance | Simpler. Nemotron streaming is a v8 stretch. It matters for sub-second latency. | Streaming vs batch ASR; how partials enable an early LLM start. | plan |
| 16 | Semantic or response caching | None | Little repeated traffic. It matters at scale with FAQ-like queries. | Cache-key design and staleness for LLM responses; prefix caching vs semantic caching. | plan |
| 17 | Online evaluation, A/B tests, drift monitoring, a user-feedback loop | Offline frozen-set eval | One user, no traffic. It matters once there are real users. | Offline/online metric mismatch; detecting drift in inputs vs outputs. | plan |
| 18 | Content moderation model | Prompt rules plus an injection suite | Single trusted user. It matters with untrusted users. | Layered guardrails: input, output and tool-level. | plan |
| 19 | Per-request cost attribution | Cost per minute measured once (v15) | Measured, not live. It matters for pricing and billing. | Cost per request from GPU-seconds and token counts. | plan |
| 20 | Load balancing across model replicas | One replica per model | Single box. It matters past N concurrent users. | Least-outstanding-requests vs round robin for LLMs; KV-cache-aware routing. | plan |
| 21 | DR targets (RPO/RTO), multi-region | Nightly dump, RPO ≈ 24 h | Personal data only. | How to choose RPO/RTO and what each costs. | plan |
| 22 | Licence compliance process | A licence column in `models.lock` | Manual review is enough for now. It matters for commercial use. MMS TTS is CC-BY-NC. | Open-weight licence types: Apache, MIT, CC-BY-NC, custom. | plan |
| 23 | Speech-to-speech models (Moshi-style) | Cascaded ASR → LLM → TTS | We need tool use, control and inspectable text. | Cascaded vs end-to-end voice: latency vs controllability. | plan |
| 24 | LLM content masked before it leaves the machine (PII scrubbing on spans) | Dev sends prompt and reply text to Langfuse when `FRYDAY_TRACE_CONTENT=true`; off by default | Masking comes in v7b. Until then, use test phrases only | Privacy by design: what to capture, opt-in content, masking at the export boundary | v0 |
| 25 | An OTel Collector between the app and the backends (batching, tail sampling, routing, scrubbing) | The app exports straight to Langfuse with the SDK's batch processor | One service and one backend, so a Collector adds a process for no gain yet | Why production routes telemetry through a Collector, and what tail sampling buys | v0 |
