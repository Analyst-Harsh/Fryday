# Experiments

Every result goes here, **including negative ones** (roadmap working principles). Newest first. A phase isn't done until its results are recorded here, in `RUNBOOK.md` or in the HLD.

## Template

```markdown
## YYYY-MM-DD: <short title> (phase N)
- **Question:** what are we trying to find out?
- **Setup:** hardware, model + revision, dataset, config, commit SHA.
- **Result:** numbers first (with units and n), then observations.
- **Decision:** what changes because of this (HLD/roadmap update, next step).
```

## 2026-10-10: Token cost of English vs Roman Hinglish vs Devanagari (F3, feeds S0-8)
- **Question:** how many LLM tokens does the same command cost in each script?
- **Setup:** `Qwen/Qwen3-4B-Instruct-2507` tokenizer, revision `cdbee75f17c01a7cc42f958dc650907174af0554`. One sentence per script (n=1). Script: `spikes/learn/f3_tokens.py`.
- **Result:**

  | Script | Tokens | Tokens per word |
  |---|---|---|
  | English | 8 | 1.14 |
  | Roman Hinglish | 12 | 1.50 |
  | Devanagari | 30 | 3.75 |

  Devanagari is 3 UTF-8 bytes per character, and byte-level BPE splits some characters across tokens.
- **Decision:** this is the first evidence for script convention **A** (Roman inside the pipeline), since it gives about 2.5× fewer LLM tokens than Devanagari. It is not decisive: n=1, and ASR accuracy (v4) and TTS quality (v3) still vote. Re-measure on more sentences in v4.

