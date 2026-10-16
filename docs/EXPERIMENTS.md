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

## 2026-10-16: Kokoro-82M on the M5: speed, memory, Hindi G2P (v3 learn 1, first look at S0-5)
- **Question:** is Kokoro fast enough for voice on the Mac, and how good is its Hindi?
- **Setup:**
  - M5, 16 GB.
  - Model: `mlx-community/Kokoro-82M-bf16`, revision `a71e4d38b236d968966a2002c4c895dbd12b1c3c` (Apache-2.0).
  - mlx-audio 0.5.9; G2P via misaki and espeak-ng (from the `espeakng-loader` wheel, no brew).
  - Voices loaded from the pinned snapshot by path. By name, mlx-audio fetches them **unpinned** from `prince-canuma/Kokoro-82M`.
  - 5 sentences, about 5 s of audio each, n=1 each, after one warm-up. Script: `spikes/learn/v3_tts_toy.py`.
- **Result:**

  | Sentence | Voice | Audio | First audio | RTF |
  |---|---|---|---|---|
  | English | af_heart | 4.88 s | 339 ms | 0.071 |
  | Hindi, Devanagari | hf_alpha | 4.78 s | 260 ms | 0.056 |
  | Hindi, Roman | hf_alpha | 4.83 s | 245 ms | 0.053 |
  | Hindi, Devanagari | hm_omega | 4.97 s | 253 ms | 0.053 |
  | Hindi, digits and ₹ | hf_alpha | 4.95 s | 251 ms | 0.053 |

  Peak MLX memory was **1.45 GB**; the weights are only about 0.16 GB.
- **Observations:**
  - **Speed:** RTF ≈ 0.05, about 18× faster than real time. First audio ≈ total time, because each sentence is one chunk; shorter first chunks (v5) should cut it.
  - **Owner's listening check:** Devanagari Hindi was "spot on"; Roman Hindi was weak.
  - **The cause was G2P, not the model.** The Hindi G2P read Roman letters with English rules: "baje" became /bˈAʤ/ ("badge"), "hai" /hˈI/, "main" /mˈAn/.
  - **Digits:** espeak expanded "7" → *saat* correctly. But **"₹250" → "do sau pachaas", with the ₹ silently dropped**. The owner rated it the best-sounding sentence: fluent, and wrong.
  - **A trap:** the library warns that EspeakG2P has no chunking, so long texts may be truncated unless split on `\n`.
- **Decision:**
  - Kokoro stays the v3 candidate. Speed is not a concern.
  - TTS input must be **Devanagari**, which supports HLD option A: transliterate before TTS. The full script decision is in v4 (S0-8).
  - The layer-2 normaliser needs a ₹ rule with a golden test (S0-9).
  - Count about 1.5 GB for TTS in the S0-2 memory budget.
  - Still to do in v3: a formal listening/MOS check, Hindi voice choice, and the MMS fallback only if needed.

## 2026-10-11: bf16 vs 8/6/4-bit Qwen3-4B on the Mac: memory, speed, agreement (v0 build 4)
- **Question:** what do 8, 6 and 4-bit quantisation buy in memory and speed on the M5, and how far do they drift from bf16?
- **Setup:**
  - Hardware: M5, 16 GB, 153 GB/s unified memory.
  - Models: `Qwen/Qwen3-4B-Instruct-2507` bf16 (revision `cdbee75`). We converted it ourselves with `mlx_lm.convert` (mlx-lm 0.32.0), affine, group 64, to 8, 6 and 4 bits.
  - Our 4-bit is **bit-identical** to `mlx-community/...-4bit` (revision `50d4277`): 904/904 tensors. Each conversion took 4–6 s.
  - 8 Hinglish prompts, greedy decoding, `max_tokens` 64, one warm-up per model. Script: `spikes/learn/v0_quant.py`.
- **Result** (n=8 prompts, single run; a second run reproduced every number within ±10 ms and ±0.2 tok/s):

  | Variant | Weights | Peak mem | TTFT | Decode | Ceiling (153 GB/s ÷ bytes) | % of ceiling | Identical replies | Tokens matched before first difference | Teacher-forced top-1 agreement |
  |---|---|---|---|---|---|---|---|---|---|
  | bf16 | 8.04 GB | 8.12 GB | 206 ms | 15.9 tok/s | 19.0 | 84% | 8/8 | 59.2 | 98.5% (noise floor) |
  | 8-bit | 4.27 GB | 4.43 GB | 165 ms | 29.4 tok/s | 35.8 | 82% | 3/8 | 36.5 | 97.3% |
  | 6-bit | 3.27 GB | 3.43 GB | 150 ms | 36.9 tok/s | 46.8 | 79% | 1/8 | 13.4 | 96.0% |
  | 4-bit | 2.26 GB | 2.43 GB | 133 ms | 53.7 tok/s | 67.6 | 79% | 0/8 | 3.1 | **77.6%** |

- **Observations:**
  - **Decode is bandwidth-bound.** Every variant reaches 79–84% of its bytes ceiling, so 3.6× fewer bytes gives 3.4× faster decode. The maths per token is the same, or slightly more for un-quantising.
  - **TTFT improves only 1.5×.** Prefill pushes about 40 tokens through each weight read, so it is partly compute-bound.
  - **Peak memory** is the weights plus 0.08–0.17 GB of KV cache and activations for short prompts.
  - **Exact match is misleading.** Greedy decoding cascades after the first near-tie flip, so even 8-bit matches only 3/8 replies. By eye, the 4-bit replies read fine.
  - **Teacher-forced agreement** (bf16's reply fed in, top-1 compared at each position) is fairer:
    - **bf16 against itself is only 98.5%**: a batched forward pass and incremental decoding round differently. That is the noise floor.
    - **8-bit and 6-bit sit within 2.5 points of the floor.**
    - **4-bit drops to 77.6%**, a clear step between 6 and 4 bits: about 1 in 4.5 next-token choices differs from bf16.
- **Decision:**
  - Keep serving 4-bit (HLD unchanged). It's 3.4× faster, the replies read fine for chat, and it fits the memory budget.
  - **6-bit is the fallback** if v1 tool-call accuracy suffers: 37 tok/s and 3.4 GB, at 96% agreement.
  - Real quality needs KL divergence and perplexity vs bf16, plus task evals (v7a). See production gap #26.

## 2026-10-10: Does mlx_lm.server honour stream_options.include_usage? (v0)
- **Question:** can we get exact token counts from a streamed reply?
- **Setup:** `mlx_lm.server`, Qwen3-4B 4-bit (MLX revision `50d4277`), openai SDK, `stream=True`, `stream_options={"include_usage": True}`, n=1.
- **Result:** **accepted.** The final chunk has `choices: []` and `usage: prompt_tokens=15, completion_tokens=7, total_tokens=22`.
- **Decision:** `llm.py` always requests usage and records it on the Langfuse generation. No fallback setting is needed.

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

