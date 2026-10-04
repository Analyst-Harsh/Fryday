# Phase 7 — Voice backends + audio

| | |
|---|---|
| **Goal** | Speech-in and speech-out models working on the Mac behind the contract, with measured baselines. |
| **Depends on** | 2, 5 |
| **Estimate** | 7–9 working days at 2–4 h/day |
| **HLD refs** | §2, §3, §3.1, §5 |
| **Status** | Not started |

## Scope
- Audio basics notes plus a feature module (resampling, log-mel) with tests.
- ASR on Triton CPU (Python backend with CT2, or ONNX). Candidates: Whisper large-v3-turbo, Oriserve Apex and Prime, and Nemotron streaming.
- ONNX parity test for the ASR encoder against PyTorch.
- Record the own-voice entity test set.
- ASR baselines: WER on the MUCS 2021 test (check speaker overlap first) and Kathbath test-unknown, plus entity accuracy. Pick the ASR and record it in `models.lock`.
- TTS: Magpie (or the fallback) in the Triton Python backend.
- Normaliser in and out, following the S0-8/S0-9 decisions, with golden tests for every rupee, number, time and date format.
- Sentence chunker (short first chunk of about 6 words).

## Out of scope
- Browser audio and turn management (phase 8).

## Exit criteria
- [ ] ASR and TTS pass the contract tests.
- [ ] Baselines are written in `EXPERIMENTS.md`.
- [ ] The ASR choice is recorded.
- [ ] Normaliser golden tests pass.

## What you learn
Audio DSP basics, ASR and TTS architectures, ONNX parity, ASR evaluation.

## Units
_Written when this phase starts. Each unit is 2–3 h and follows the `superpowers:writing-plans` task format (files, failing test, implementation, verification, commit)._
