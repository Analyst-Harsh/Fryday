# Phase 12 — GPU sprint 1: fine-tuning

| | |
|---|---|
| **Goal** | Fine-tune the LLM and ASR, quantise them, and promote only what passes the gate. |
| **Depends on** | 10, 11 |
| **Estimate** | 5–7 working days at 2–4 h/day |
| **HLD refs** | §3, §4, §5, §11 (M3, M5) |
| **Status** | Not started |

## Scope
- First hour: generate the full synthetic LLM dataset with the phase 10 pipeline.
- Minimal vLLM bring-up behind the SSH tunnel, needed for the gate and E1; it is hardened in phase 13.
- LLM LoRA SFT, merge, then quantise to AWQ, GPTQ and MLX 4-bit.
- Run the gate for each (model, backend, format) row; promote passing artefacts in `models.lock`.
- E1: constrained decoding vs fine-tune vs both.
- E2: quality and speed across the quantisation formats.
- ASR LoRA, trained after the LLM; evaluated on the Mac or GPU against the frozen ASR sets.

## Out of scope
- GPU serving optimisation (phase 13).

## Exit criteria
- [ ] Fine-tuned LLM artefacts are scored against the thresholds in `eval/gate.yaml`, and promoted only if they pass.
- [ ] ASR entity accuracy and the WER change versus the phase 7 baseline are reported with confidence intervals.
- [ ] The ASR LoRA is trained and evaluated.
- [ ] E1 and E2 are written up.
- [ ] GPU hours and cost are recorded.

## What you learn
LoRA/PEFT, SFT, quantisation in practice.

## Units
_Written when this phase starts. Each unit is 2–3 h and follows the `superpowers:writing-plans` task format (files, failing test, implementation, verification, commit)._
