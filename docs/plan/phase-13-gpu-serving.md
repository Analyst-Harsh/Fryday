# Phase 13 — GPU sprint 2: serving + optimisation

| | |
|---|---|
| **Goal** | Serve everything from the GPU, then make it fast and understand why. |
| **Depends on** | 12 |
| **Estimate** | 6–8 working days at 2–4 h/day |
| **HLD refs** | §3, §4, §11 (M4, M5) |
| **Status** | Not started |

## Scope
- GPU Triton for ASR, TTS and embeddings, plus vLLM for the LLM; the e2e and contract suites pass on the GPU.
- E4: PyTorch vs ONNX vs TensorRT on embeddings.
- E6: CUDA graphs on vs off.
- E7: one Triton instance vs separate servers.
- E8: Triton's vLLM backend vs standalone vLLM.
- Nsight Systems: an annotated timeline of one request. Grafana and DCGM run for the session.
- M5 ASR gate.
- Stretch, time-boxed: TRT-LLM Whisper encoder (E5).

## Out of scope
- Load testing (phase 14).

## Exit criteria
- [ ] All models are served from the GPU.
- [ ] E4 and E6–E8 are written up.
- [ ] The Nsight timeline is recorded.
- [ ] The M5 gate result is recorded.

## What you learn
TensorRT, Triton in depth, CUDA working principles.

## Units
_Written when this phase starts. Each unit is 2–3 h and follows the `superpowers:writing-plans` task format (files, failing test, implementation, verification, commit)._
