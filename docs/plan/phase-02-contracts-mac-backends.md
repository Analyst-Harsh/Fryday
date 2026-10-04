# Phase 2 — Contracts + Mac backends

| | |
|---|---|
| **Goal** | The app can call every model through the two contracts on the Mac, with tests. |
| **Depends on** | 1 |
| **Estimate** | 5–6 working days at 2–4 h/day |
| **HLD refs** | §1.3, §2 (warm-up), §3; architecture doc §6, §10 |
| **Status** | Not started |

## Scope
- `backends` module: a KServe v2 gRPC client (tritonclient) and an OpenAI-compatible client; `MODEL_BACKEND` config.
- Stub servers for both contracts, for use in CI.
- Contract tests parametrised by backend URL, including tool-call JSON-schema and chat-template parity checks (HLD §10).
- Triton CPU model repository with `embed` (bge-m3 exported to ONNX), or the ORT fallback chosen in S0-1.
- MLX-LM server launch script (runs natively on the host).
- Readiness checks with a 1–2 s probe timeout, plus a warm-up inference before reporting ready.

## Out of scope
- ASR and TTS models (phase 7); the GPU.

## Exit criteria
- [ ] Contract tests pass against the stubs in CI and against the real Mac backends locally.
- [ ] A Hinglish sentence is embedded over gRPC.
- [ ] A chat completion streams from MLX.
- [ ] Readiness stays false until warm-up has finished.

## What you learn
The KServe v2 protocol, gRPC, Triton model repositories, exporting an encoder to ONNX.

## Units
_Written when this phase starts. Each unit is 2–3 h and follows the `superpowers:writing-plans` task format (files, failing test, implementation, verification, commit)._
