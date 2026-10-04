# Phase 6 — LLM eval harness + baselines

| | |
|---|---|
| **Goal** | Measure the LLM before changing it. |
| **Depends on** | 4, 5 (the eval covers grocery tool calls, which need the mock's schemas) |
| **Estimate** | 6–8 working days at 2–4 h/day |
| **HLD refs** | §5 |
| **Status** | Not started |

## Scope
- Frozen set of 200 Hinglish tool conversations covering every tool, grocery included. They are drafted with the local LLM from hand-written seeds, and **every item is hand-reviewed**. The set has dedup and template-overlap checks and a usage counter.
- Metrics: tool-choice accuracy, argument validity, and a judge score. The judge is a different model from the generator, calibrated on a human-labelled subset.
- Gate-matrix code: one row per (model, backend, format), with confidence intervals and a promotion margin; plus the Mac-vs-GPU divergence metric.
- `models.lock` format, with the first entry for the current LLM. Backend launch scripts read `models.lock` and refuse unpinned artefacts.
- **Gate thresholds and the promotion margin are committed to `eval/gate.yaml` before phase 12**, so promotion can be falsified.
- Baselines: Qwen3-4B on MLX, and a hosted model run offline.

## Out of scope
- Fine-tuning; ASR evaluation (phase 7).

## Exit criteria
- [ ] One command produces a scored report.
- [ ] Qwen and hosted baselines are in `EXPERIMENTS.md`.
- [ ] Judge–human agreement on the labelled subset is at or above the threshold committed in `eval/gate.yaml`, and the value is recorded in `EXPERIMENTS.md`.
- [ ] `eval/gate.yaml` holds per-row thresholds and the promotion margin.
- [ ] `models.lock` pins the served LLM.

## What you learn
LLM evaluation, LLM-as-judge calibration, confidence intervals, data hygiene.

## Units
_Written when this phase starts. Each unit is 2–3 h and follows the `superpowers:writing-plans` task format (files, failing test, implementation, verification, commit)._
