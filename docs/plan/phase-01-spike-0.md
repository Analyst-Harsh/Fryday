# Phase 1 — Spike 0 (Mac)

| | |
|---|---|
| **Goal** | Answer the risky Mac-side questions before building on them. |
| **Depends on** | 0 |
| **Estimate** | 4–6 working days at 2–4 h/day |
| **HLD refs** | §11 (M0), architecture doc §17 |
| **Status** | Not started |

## Scope
- **S0-1:** does Triton's arm64 CPU container serve an ONNX model over KServe v2 gRPC on the M5? If not, plain ONNX Runtime goes behind the same contract.
- **S0-2:** does the `core` profile plus MLX-LM fit in 16 GB without swapping?
- **S0-3:** does `mlx_lm.server` stream, and does Qwen3-4B produce valid tool calls through it?
- **S0-5:** confirm exact model IDs and licences. Also check that Magpie produces intelligible Hindi on the Mac CPU (fallback: `facebook/mms-tts-hin`).
- **S0-8:** choose the script convention, A (Roman inside) or B (Devanagari everywhere), by testing on public dataset clips.
- **S0-9:** convert Hindi numbers to words. Check whether NeMo or `num2words` passes the golden cases, or plan custom rules.
- **Setup items:** Hugging Face token, dataset downloads and licence check, Langfuse signup.
- Spike code lives in `spikes/` and is labelled throwaway.

## Out of scope
- Production code; GPU spikes (phase 11).

## Exit criteria
- [ ] Every spike has a pass/fail result and a decision recorded in `EXPERIMENTS.md`, against these thresholds: S0-2 peak memory ≤ 13 GB with no swap growth over a 10-minute loop; S0-3 at least 18 of 20 streamed tool calls valid; S0-5 Magpie's Hindi real-time factor recorded and a listening check passed.
- [ ] Quick checks recorded: Nemotron streaming on a few Hinglish clips, MUCS test speaker overlap, and whether IndicXlit is available.
- [ ] The HLD is updated wherever a decision changed it (for example ORT instead of Triton on the Mac, script convention, numbers→words approach, TTS choice).

## What you learn
Triton model repository and config.pbtxt basics, MLX, model licensing, Hinglish script problems.

## Units
_Written when this phase starts. Each unit is 2–3 h and follows the `superpowers:writing-plans` task format (files, failing test, implementation, verification, commit)._
