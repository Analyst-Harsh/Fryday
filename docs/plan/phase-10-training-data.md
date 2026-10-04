# Phase 10 — Training data + dry runs

| | |
|---|---|
| **Goal** | Get the datasets ready and prove the training scripts on the Mac, so GPU time isn't wasted. |
| **Depends on** | 5, 6, 7 |
| **Estimate** | 6–8 working days at 2–4 h/day |
| **HLD refs** | §3, §4, §5 |
| **Status** | Not started |

## Scope
- Synthetic LLM tool-conversation data (5–10k examples) from an open-weight generator. Every example is validated against the mock tools, then deduplicated against the frozen set.
- ASR data: training manifests from IndicVoices Hindi, Kathbath train and MUCS train, plus a held-out test corpus not used in training, with speaker overlap checked.
- DVC for datasets, MLflow for runs.
- LoRA training scripts run for a few steps on the Mac (MLX, or tiny CPU runs).
- E3: a from-scratch int8/int4 quantiser with error analysis.
- KV-cache memory worked out on paper for the chosen LLM.

## Out of scope
- Real training runs (phase 12).

## Exit criteria
- [ ] The datasets are versioned in DVC with lineage.
- [ ] Training scripts complete end to end on the Mac.
- [ ] E3 is written up.
- [ ] The KV-cache calculation is recorded.

## What you learn
Data pipelines, synthetic data quality, quantisation theory, KV cache.

## Units
_Written when this phase starts. Each unit is 2–3 h and follows the `superpowers:writing-plans` task format (files, failing test, implementation, verification, commit)._
