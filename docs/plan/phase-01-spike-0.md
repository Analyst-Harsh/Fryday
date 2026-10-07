# Phase 1 — Spike 0 (Mac)

| | |
|---|---|
| **Goal** | Answer the risky Mac-side questions before building on them. |
| **Depends on** | 0 |
| **Estimate** | 4–6 working days at 2–4 h/day |
| **HLD refs** | §11 (M0), architecture doc §17 |
| **Status** | Superseded (2026-10-07): the spikes are folded into versions v0–v12 of the [learning roadmap](00-learning-roadmap.md) (see its spike table). The decisions below still stand. |

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

**Decisions (2026-10-05):**
- Golden cases: drafted by Claude, reviewed by the owner. Spellings use रुपये, and हज़ार with the nukta.
- Datasets: download only what the spikes need, which is the MUCS 2021 Hi-En test set (OpenSLR 104) and about 20 FLEURS `hi_in` clips. Accept the terms for IndicVoices, Kathbath and Shrutilipi now; download them in phase 10.
- S0-8 is scored on 50 MUCS clips in both scripts, plus a TTS listening check on about 10 sentences per option.
- Langfuse Cloud uses the **EU** region.

### Conventions
- One branch, `phase-1-spike-0`, with a commit per unit and one PR at the end, the same as phase 0.
- Spike code lives in `spikes/s0_N_<name>/` and is labelled throwaway.
  - Each script declares its own dependencies with **PEP 723 inline metadata** and runs with `uv run spikes/.../x.py`.
  - Heavy dependencies (mlx, nemo, torch, tritonclient) stay out of the workspace `uv.lock`.
  - ruff and pyright already exclude `spikes/`.
- Weights and datasets go in a git-ignored `data/` folder (add it to `.gitignore`) or the HF cache.
- Spikes are **measured, not asserted**. The exception is S0-9, which gets a golden test runner.
- Each unit writes its own `EXPERIMENTS.md` entry using the existing template, with model revision SHAs.


| # | Unit | Est. | Owner hands-on time | Needs |
|---|---|---|---|---|
| U1 | Setup, accounts, golden draft | 2.5 h | ~45 min (logins, signup) | — |
| U2 | S0-5 IDs and licences, MUCS overlap, prefetch | 2 h | — | U1 |
| U3 | S0-1 Triton arm64 CPU | 3 h | — | U2 |
| U4 | S0-3 MLX streaming and tool calls | 2.5 h | — | U2 |
| U5 | S0-5 Magpie Hindi plus Nemotron check | 3 h (timeboxed) | ~20 min of listening | U2 |
| U6 | Owner reviews the golden cases | — | ~45 min | U1 |
| U7 | S0-9 numbers→words scoring | 2 h | — | U6 |
| U8 | S0-8a ASR in both scripts, IndicXlit, scoring | 3 h | ~20 min of entity tagging | U2 |
| U9 | S0-8b TTS listening, then decide A or B | 1.5 h | ~20 min of listening | U5, U8 |
| U10 | S0-2 memory fit | 3 h | — | U3, U4, U5 |
| U11 | Close-out: HLD updates and PR | 1.5 h | PR review | all |

That totals about 24 h of Claude work plus about 3 h of owner time, which lands at the top of the 4–6-day estimate. **Critical path:** U1 → U2 → U5 → U9, with U10 last. U3, U4, U7 and U8 can go in any order around it.

### Unit U1 — Setup, accounts, golden draft
1. ~~Paste these units into this file and set Status → In progress.~~ Done 2026-10-05.
2. **Owner:** install the CLI with `uv tool install "huggingface_hub[cli]"`, then run `hf auth login`, so the token never enters the chat. Accept the HF terms for IndicVoices, Kathbath and Shrutilipi. Sign up for Langfuse in the EU region and create a project.
3. Add these placeholders to `.env.example`: `HF_TOKEN`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST=https://cloud.langfuse.com`.
4. Add `data/` to `.gitignore`. Download the MUCS test set and about 20 FLEURS `hi_in` clips, and record their licences.
5. Write `spikes/README.md`: a "throwaway" note plus the PEP 723 convention.
6. Draft `spikes/s0_9_numbers/golden.tsv`, about 40 cases of `input → expected Hindi words`. It covers ₹ amounts and paise, the lakh/crore grouping, decimals, time, date, ordinals, units and plain numbers. Hand it to the owner for U6.
7. Commit.

### Unit U2 — S0-5 IDs and licences, MUCS overlap, prefetch
1. For every HLD §3 candidate, record the exact ID, revision SHA, licence and whether it is gated:
   - Qwen3-4B-Instruct-2507, plus its `mlx-community` 4-bit build;
   - whisper-large-v3-turbo;
   - Oriserve Apex and Prime;
   - Nemotron 3.5 streaming;
   - Magpie;
   - `facebook/mms-tts-hin`;
   - bge-m3 and e5-small.
2. Settle the MMS licence, which is marked *(verify)*.
3. Check whether MUCS test speakers overlap with train speakers, using the metadata speaker IDs.
4. Prefetch the weights the later units need.
5. Write the EXPERIMENTS entries and commit.

### Unit U3 — S0-1 Triton arm64 CPU over KServe v2 gRPC
1. Pull the current `nvcr.io/nvidia/tritonserver:<ver>-py3`. Confirm it has an arm64 manifest, and pin it by digest.
2. Build a model repo with two models:
   - `multilingual-e5-small` as ONNX (from the repo's `onnx/` folder, or exported with optimum), with `backend: "onnxruntime"` and `KIND_CPU`;
   - a trivial **Python-backend** model.
3. Write a client with `tritonclient[grpc]` `ModelInfer`. It checks cosine ≥ 0.999 against local ORT on the same input.
4. Record image size, RSS, latency, and pass or fail. If it fails, the decision is "ORT behind the KServe v2 contract", built in phase 2.

### Unit U4 — S0-3 MLX streaming and tool calls
1. Start `mlx_lm.server` with Qwen3-4B 4-bit. First check that plain-text streaming works without tools.
2. Run 20 Hinglish prompts with 4 tool schemas (calendar, search, notes, reminder) and `stream=True`. Accumulate the `tool_calls` deltas and validate them with Pydantic.
3. Record the valid count, TTFT and tokens per second. **Pass: at least 18 of 20.** If it fails, record which way it fails (no parsing, broken JSON, wrong tool). That feeds phase 3's validate-and-repair step.

### Unit U5 — Magpie Hindi on CPU, plus the Nemotron check (both in one NeMo environment)
1. Install NeMo, with a 1 h timebox.
2. Synthesise 10 Hindi and Hinglish sentences with Magpie on CPU, and record the RTF.
3. **Owner** listen and mark each sentence pass or fail.
4. If Magpie fails, try `facebook/mms-tts-hin`, provided the U2 licence check allows it.
5. Nemotron streaming: transcribe about 5 MUCS clips and record the outputs. There is no threshold.

### Unit U6 — Owner reviews the golden cases
Correct `golden.tsv` in place, or leave comments. This happens in parallel with U2–U5.

### Unit U7 — S0-9 numbers→words scoring (test-first)
1. The runner scores `num2words(lang="hi")` and NeMo `Normalizer(lang="hi")`, if one exists, against every golden case. Installing `nemo_text_processing` (pynini) gets a 1 h timebox.
2. Decide: adopt a library only if it passes **all** cases; otherwise write custom rules in phase 2 or 7. The TSV stays as the future golden test.

### Unit U8 — S0-8a ASR in both scripts
1. Get IndicXlit working: `ai4bharat-transliteration` on Python 3.12, falling back to an older Python. That also answers the exit criterion's IndicXlit quick check.
2. Run the same 50 MUCS clips through the transformers pipeline on CPU or MPS:
   - Oriserve Apex, which outputs Roman script (option A);
   - whisper-large-v3-turbo with `language=hi`, which outputs Devanagari (option B).
3. Score both in Devanagari against the references: A goes through IndicXlit first, then WER and CER are computed with `jiwer`.
4. **Owner** tag about 10 clips that contain entities or numbers, and I score entity accuracy on those clips.

### Unit U9 — S0-8b TTS check, then decide A or B
1. **Owner** listen to two sets through Magpie (or the U5 fallback): 10 Roman Hinglish sentences sent through IndicXlit, and 10 native Devanagari sentences.
2. Combine this with the U8 numbers, and record the decision.

### Unit U10 — S0-2 memory fit
1. Convert whisper-large-v3-turbo to CT2 int8 with `ct2-transformers-converter`.
2. Bring everything up together:
   - `make up` (Postgres);
   - Triton with e5 ONNX (or ORT, if U3 failed);
   - one process holding CT2 whisper and Magpie;
   - `mlx_lm.server` with Qwen3-4B.
3. Run a loop of fake turns for **10 minutes**: ASR clip → embed → ~100 LLM tokens → one TTS sentence.
4. Sample `vm_stat`, `sysctl vm.swapusage` and `memory_pressure` every 5 s. The OrbStack VM counts toward the total.
5. **Pass: peak used ≤ 13 GB and no swap growth.** If it fails, shrink one thing (e5-small, or a smaller ASR) and re-run once.

### Unit U11 — Close-out
1. Update the HLD with every decision: §3, §3.1, the Decisions table and "Still unverified".
   - Examples: ORT vs Triton, A vs B, the numbers approach, the TTS choice, licence findings.
   - Add Magpie-in-Triton-Python-backend as a residual risk.
2. Update architecture §17.
3. Tick the exit criteria, set Status → Done and open the PR.
