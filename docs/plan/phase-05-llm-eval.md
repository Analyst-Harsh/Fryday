# Phase 5 — LLM eval harness + baselines

| | |
|---|---|
| **Goal** | Measure the LLM before changing it. |
| **Depends on** | 4 |
| **Estimate** | 6–8 working days at 2–4 h/day |
| **HLD refs** | §5 |
| **Status** | Not started |

## Scope
- Frozen set of 200 hand-reviewed Hinglish tool conversations (seeds, generation, review), with dedup and template-overlap checks and a usage counter.
- Metrics: tool-choice accuracy, argument validity, and a judge score. The judge is a different model from the generator, calibrated on a human-labelled subset.
- Gate-matrix code: one row per (model, backend, format), with confidence intervals and a promotion margin; plus the Mac-vs-GPU divergence metric.
- `models.lock` format, with the first entry for the current LLM.
- Baselines: Qwen3-4B on MLX, and a hosted model run offline.

## Out of scope
- Fine-tuning; ASR evaluation (phase 7).

## Exit criteria
- [ ] One command produces a scored report.
- [ ] Qwen and hosted baselines are in `EXPERIMENTS.md`.
- [ ] Judge agreement meets the threshold.
- [ ] `models.lock` pins the served LLM.

## What you learn
LLM evaluation, LLM-as-judge calibration, confidence intervals, data hygiene.

## Units
_Written when this phase starts. Each unit is 2–3 h and follows the `superpowers:writing-plans` task format (files, failing test, implementation, verification, commit)._
