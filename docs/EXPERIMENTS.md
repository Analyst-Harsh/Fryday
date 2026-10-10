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

## 2026-10-10: Repetition loops vs sampling config (v0, inconclusive)
- **Question:** the first live `fryday-chat` run produced one runaway loop: one line repeated until `max_tokens`, giving 484 chunks. Does the sampling config cause it?
- **Setup:**
  - Model: Qwen3-4B 4-bit, MLX revision `50d4277`, served by `mlx_lm.server`.
  - Context: a fixed 2-turn Hinglish conversation.
  - Sampling: temperature 0.7, `max_tokens` 300, 6 runs per config.
  - Configs compared: our original (top_p 0.9, no top_k), Qwen's recommended (top_p 0.8, top_k 20), and recommended plus presence_penalty 1.0.
  - Script: `spikes/learn/v0_repetition.py`.
- **Result:** **0/6 loops under every config.** Reply lengths were 30–69 tokens. The live loop came after an unusual first reply, so it is rare and depends on context.
- **Decision:**
  - Adopt Qwen's published defaults (top_p 0.8, top_k 20). This follows the vendor's documentation; the measurement did not show they are better.
  - Keep the `max_tokens` cap as the hard guard.
  - Don't add presence_penalty without evidence.
  - Follow-ups:
    - Lower the cap for voice replies (v5).
    - Measure loop rate at scale in the v7a eval, which needs n ≫ 6.

## 2026-10-10: Qwen3-4B 4-bit on the M5 via MLX: speed and memory (F7, feeds S0-2 and S0-3)
- **Question:** what time to first token, decode speed and memory do we get on the Mac, and does `mlx_lm.server` stream?
- **Setup:**
  - Hardware: M5, 16 GB.
  - Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit`, revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b` (2.28 GB on disk).
  - Runs: `mlx_lm.generate`, then `mlx_lm.server` called through the OpenAI API with `stream=True`.
  - Prompt: a system prompt plus one Hinglish user turn, generating about 60 tokens.
  - Requests: 3 sequential runs, n=1 each.
  - Client: `spikes/learn/f7_probe.py`.
- **Result:**
  - `generate`: decode **54.9 tok/s**, peak memory **2.40 GB**.
  - Server TTFT by run:

    | Run | TTFT | Note |
    |---|---|---|
    | 1 | 0.53 s | warm-up |
    | 2 | 0.11 s | |
    | 3 | 0.05 s | identical prompt, so the prompt cache was reused |

  - Server decode: **about 42 tok/s**, steady across runs.
  - Server memory: RSS **2.36 GB**, physical footprint **2.6 GB**.
  - Streaming over SSE works (the plain-text half of S0-3).
- **Observations:**
  - The first request pays a warm-up cost. That supports the HLD rule: warm up before reporting ready.
  - Decode at about 42 tok/s is well above speaking pace (about 4–5 tok/s), so time to first token matters more for voice.
  - The model claimed "Reminder set!" with no tool attached. That is the motivation for v1's tool calling and validation.
- **Decision:**
  - The LLM fits the Mac budget comfortably at about 2.6 GB; S0-2 still has to include ASR, TTS and Postgres.
  - The tool-call half of S0-3 moves to v1.

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

