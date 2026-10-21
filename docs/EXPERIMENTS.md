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

## 2026-10-21: asyncio pipeline toy: backpressure, barge-in, and the blocking trap (v5 learn 1)
- **Question:** in an async voice pipeline, how fast does barge-in stop the audio, and what does a blocking model call do to the event loop?
- **Setup:** `spikes/learn/v5_asyncio_toy.py`, all fakes:
  - an LLM producing a word every 30 ms → chunker → TTS (200 ms `time.sleep` standing in for Kokoro) → audio queue (`maxsize=1`) → speaker (1 s per chunk);
  - a mic ticking every 50 ms;
  - barge-in at 1.4 s (moved from 1.0 s so it lands during a TTS call).
  - Run A: TTS via `asyncio.to_thread`. Run B: TTS called directly.
- **Prediction (owner):** audio stops after the current TTS call finishes.
- **Result:**

  | | A: `to_thread` | B: direct (blocking) |
  |---|---|---|
  | Mic stalls | none | ~230 ms on every TTS call (4×) |
  | Barge-in timer | 1 ms late | **134 ms late** |
  | Audio stops after `cancel()` | 0.5 ms | 0.1 ms (but the cancel itself came late) |
  | TTS thread after cancel | kept running 140 ms; result discarded | — |
  | Backpressure (queue full, TTS waits) | yes, at 0.74 s | yes |
  | Queued chunks left to drop | 1 | 1 |

- **Findings:**
  - With `to_thread`, cancellation hits the task at its `await`, so the audio stops instantly. The prediction was wrong here.
  - The **thread can't be cancelled**: the compute finishes and is wasted (GPU time the next reply waits for).
  - A direct blocking call freezes the whole loop. The mic and VAD are starved, and barge-in is noticed late; there, the prediction is exactly right.
- **Decision for v5:**
  - Every in-process model call (Whisper, Kokoro, VAD) goes through `to_thread`.
  - On barge-in, cancel the reply task **and** drop the queued audio.
  - Keep TTS chunks short to limit orphaned compute.

## 2026-10-21: Where does "first token → first audio" go? The latency budget (F9 toy)
- **Question:** of the ~1.3 s between the LLM's first token and the first audio, how much is waiting for the first full sentence and how much is Kokoro?
- **Setup:**
  - `spikes/learn/f9_budget.py`.
  - `mlx_lm.server` with Qwen3-4B 4-bit, warm, plus Kokoro `hf_alpha` in-process, warm.
  - 5 new Hindi questions with the voice system prompt, after one warm-up question that is dropped.
  - Each first sentence (from `sentences()`) goes through `normalise()`, then the full `tts.speak()`. Kokoro returns a whole sentence at once.
- **Prediction (owner):** mostly Kokoro.
- **Result** (ms, median of 5): time to first token 334 · **A. sentence aggregation 844** · **B. Kokoro 397** · first token → first audio **1,241**.

  | Question | First token | A. Aggregation | B. Kokoro | Words | Audio |
  |---|---|---|---|---|---|
  | आज मौसम कैसा है? | 339 | 851 | 391 | 9 | 4.0 s |
  | मुझे कल सुबह सात बजे याद दिलाना। | 265 | 705 | 363 | 7 | 3.6 s |
  | एक अच्छी किताब बताओ। | 457 | 768 | 397 | 8 | 3.8 s |
  | चाय बनाने का आसान तरीका बताओ। | 334 | 1,663 | 594 | 21 | 7.6 s |
  | तुम कौन हो? | 115 | 844 | 454 | 8 | 4.1 s |

- **Findings:**
  - The prediction was wrong. **Aggregation is 68%**: about 105 ms per Devanagari word (~3.75 tokens per word at ~36 tokens/s), so a 21-word first sentence costs 1.7 s.
  - Kokoro is about 10× faster than real time, with ~300 ms of fixed overhead (8 words ≈ 400 ms, 21 words ≈ 600 ms).
  - The cold-prompt time to first token is ~330 ms. The 90 ms in the 2026-10-19 entry reused a cached prompt.
  - Budget compared with the voiceaiandvoiceagents.com primer (§5.1: about 1,300 ms typical, 1,500 ms target, 500 ms human turn gap):

    | Stage | Primer | Fryday |
    |---|---|---|
    | ASR | — | 450 ms |
    | Time to first token | 650 ms | 330 ms |
    | Aggregation + TTS | 140 ms | 1,240 ms |
    | Endpointing and network | — | not yet (push-to-talk, local) |
    | **Total** | ~1,300 ms | **≈ 2.0 s** |

- **Decision for v5, by impact:**
  1. A short first chunk (~4 words), with aggregation expected to fall from ~850 ms to ~400 ms.
  2. Measure Roman replies (~2.5× fewer tokens).
  3. Leave Kokoro as is.

## 2026-10-20: Script convention, Whisper → Devanagari vs Oriserve → Roman Hinglish (v4 build, S0-8)
- **Question:** which ASR output script should Fryday use? Is the Hinglish fine-tune more accurate than off-the-shelf Whisper?
- **Setup:**
  - `spikes/learn/v4_script_eval.py`.
  - 100 random clips (seed 0) from MUCS-Hinglish test shard 0 (`dianavdavidson/MUCS-Hinglish` @ `3c66ee1`, CC-BY-4.0; 1.3–30.6 s, median 5.5 s; read technical tutorials).
  - Whisper large-v3-turbo via mlx-whisper (`a4aaeec`, `language="hi"`, default loop guard).
  - Oriserve `Whisper-Hindi2Hinglish-Apex` (turbo base, Apache-2.0, `f3214ee`) via transformers on MPS, `language="en"` as its card says, long-form timestamps on.
  - **Score:** references are mixed script, so every text is reduced to a sound skeleton (IAST romanisation, accents, vowels and h stripped, repeats collapsed), then CER. One run per configuration.
- **Prediction (owner):** Whisper with Hindi wins.
- **Result:**

  | Oriserve config | Whisper median CER | Oriserve median CER | Clips won (W / O) | s/clip (W / O) |
  |---|---|---|---|---|
  | fp16, no loop guard | 30.4% | 47.7% | 66 / 29 | 0.51 / 1.35 |
  | fp32, no loop guard | 30.4% | 47.7% | 66 / 29 | 0.51 / 3.14 |
  | fp16, **same loop guard as mlx-whisper** (temperature fallback, compression ratio 1.35 in token space, logprob −1.0, no conditioning) | 31.1% | 31.3% | 56 / 37 | 0.50 / 1.71 |

- **Findings:**
  - **Without the guard, Oriserve looped** (`nan?nan?…`, `0.0.0…`); pooled CER was 232%. fp32 didn't change it, so it wasn't precision. With the guard it **tied** Whisper. Most of the 16-point "loss" came from our harness, not the model.
  - Oriserve still emits stray junk tokens: a trailing `nan` (likely broken training labels; hypothesis) and Cyrillic `Кор`.
  - Whisper's score moved between identical runs (pooled 34% → 41%, wins 66 → 56) because its fallback samples randomly. n = 1 run per configuration, so gaps of a few points are noise.
  - MUCS audio and references are partly misaligned: both models agree with each other but not with the reference. Absolute CERs are inflated; only the comparison means something.
  - Whisper with `language="hi"` sometimes writes English words in Latin script ("document", "replace with field").
  - Speed isn't a fair comparison (MLX vs PyTorch MPS). Oriserve is turbo-based, so an MLX conversion would likely match Whisper.
- **Decision:**
  - **ASR stays Whisper with Devanagari**: equal accuracy, faster in our stack, no junk tokens. HLD §3.1 updated.
  - The token cost of Devanagari sits in the LLM **reply**, not the ASR input. Roman replies plus transliteration before TTS is now a v5 latency measurement.
- **Lesson:** match decoding settings before comparing models. A fine-tune isn't automatically better off its own data distribution.

## 2026-10-20: Inside Whisper: padding cost, the log-mel grid, silence (v4 learn 1 toy)
- **Question:** does a 3 s clip cost as much as 30 s (padding)? What does Whisper output on silence?
- **Setup:** `spikes/learn/v4_asr_toy.py`. Whisper large-v3-turbo (mlx-whisper, `a4aaeec`, `language="hi"`) through `ASRClient`, warm, median of 3 runs. Input: the owner's 3 s F8 clip, repeated with `np.resize` to 10 s and 30 s; 5 s of zeros; 5 s of Gaussian noise (σ = 30, about −60 dBFS).
- **Predictions (owner):** same time for 3 s and 30 s; invents words on silence.
- **Result:**

  | Input | Time | Output |
  |---|---|---|
  | 3 s speech | 0.45 s | फ्राइडे, कल सुबह साथ बज़े याद देखे |
  | 10 s (clip repeated) | 0.44 s | the phrase **once** |
  | 30 s (clip repeated) | 0.44 s | the phrase **once** |
  | 5 s pure silence | — | **झाल झाल** |
  | 5 s quiet noise | — | **झाल झाल** |

  Log-mel: 3 s → (300, 128), i.e. 100 frames/s (hop 160 at 16 kHz) × 128 mel bins. Padded to (3000, 128) for the encoder.
- **Findings:**
  - Both predictions held. Hallucination on silence is real, which justifies the `SILENT_PEAK` gate in `talk.py`.
  - **The timing test is confounded.** Whisper transcribed repeated speech only once, even with 0.5 s gaps (21 s). The decoder did the same work for every length, so this only shows that the encoder cost is fixed (always 3,000 frames). A clean test needs 30 s of varied speech.
  - Warm ASR is **0.45 s**, not the ~2 s from the 2026-10-19 n=1 smoke run. ASR is not the main latency cost; the first full sentence (~1.3 s) is.

## 2026-10-20: Which Kokoro Hindi voice? Blind listening (v3 build 3, half of S0-5)
- **Question:** which of Kokoro's 4 Hindi voices sounds best for Fryday?
- **Setup:** `spikes/learn/v3_voice_pick.py`. The same 3 sentences (greeting, a 7:30 time, ₹2.5 लाख and ₹15,999), each passed through `normalise()`, in `hf_alpha`, `hf_beta`, `hm_omega` and `hm_psi`. The files were saved under shuffled letters A–D, with the key hidden until the owner chose. One listener (the owner), n=1, no scores, just a preference.
- **Prediction:** the owner expected to prefer `hf_beta`.
- **Result:** picked **D = `hf_alpha`** as the most natural (A = `hf_beta`, B = `hm_omega`, C = `hm_psi`). All numbers were heard correctly (साढ़े सात बजे, ढाई लाख रुपये, पंद्रह हज़ार नौ सौ निन्यानबे रुपये), the first end-to-end ear check of `normalise()`.
- **Decision:** keep `tts_voice = "hf_alpha"`. The prediction was wrong, which is the reason to listen blind. One listener isn't a MOS; a multi-listener test comes with v7a evals if voice quality becomes a question.

## 2026-10-20: What does Kokoro's text frontend read correctly? Hindi (espeak) vs English (misaki) (v3 build 2, S0-9)
- **Question:** which number, money and time forms does the TTS frontend already speak right, so the rules layer only fixes the rest?
- **Setup:** misaki 0.9 `EspeakG2P("hi")` (what mlx-audio Kokoro uses for `lang_code="h"`) and `misaki.en.G2P` (US English), phonemes read by hand. n=1 per input, deterministic.
- **Result:**

  | Input | Hindi (espeak-ng) | English (misaki) |
  |---|---|---|
  | `99` / `150000` | निन्यानबे / एक लाख पचास हज़ार ✅ | ✅ |
  | `50%` | पचास प्रतिशत ✅ | fifty percent ✅ |
  | `2.5` | दो . पाँच ❌ | two point five ✅ |
  | `7:30` | सात तीस ❌ | seven ❓ thirty ⚠️ |
  | `₹250` | दो सौ पचास, **₹ dropped** ❌ | ❓ two fifty ❌ |
  | `1,50,000` | एक, पचास, शून्य शून्य शून्य ❌ | one hundred fifty thousand ✅ |
  | `10/12` | दस स्लैश बारह ❌ | ten twelve ⚠️ |
  | `3rd` | तीन आरडी ❌ | third ✅ |

- **Findings:**
  - misaki has real text normalisation only for English (`num2words`, `$ £ €`). For Hindi it passes text straight to espeak-ng.
  - espeak reads plain integers and % well, including lakh grouping, so build 2 rewrites only ₹, times, decimals and Indian commas, and leaves digits to espeak (`app/src/fryday/normalise.py`, 21 golden tests).
  - This explains the missing रुपये in the 2026-10-18 `hi_digits` result.
- **Decision:** regex rules now; dates and ordinals are left out (rare in a tool-less chatbot). Relying on espeak for digits couples us to the TTS frontend (production gap #31).

## 2026-10-19: First speech-to-speech turn on the Mac: latency by stage (v3 build 1 skeleton)
- **Question:** does mic → ASR → LLM → TTS work end to end on the M5, and where does the time go?
- **Setup:**
  - M5, 16 GB. `mlx_lm.server` with Qwen3-4B 4-bit (separate process); Whisper large-v3-turbo (mlx-whisper, revision `a4aaeec`, `language="hi"`) and Kokoro-82M (mlx-audio, `hf_alpha`) in-process.
  - Input: the owner's 3 s F8 recording (`data/audio/f8_16k.wav`) instead of the live mic. Voice system prompt (Devanagari, numbers as words). Sentence chunking, no playback.
  - n=1 cold + 1 warm run, same input. Script: a scratchpad smoke script over `ASRClient`, `LLMClient`, `TTSClient`, `speak_reply`.
- **Result:**

  | Run | ASR | LLM first token | First audio (from ASR done) | You stop → Fryday speaks | Reply audio |
  |---|---|---|---|---|---|
  | cold | 1.89 s | 8.49 s | 11.28 s | 13.2 s | 10.6 s, 3 sentences |
  | warm | 2.14 s | 0.09 s | 1.42 s | **3.6 s** | 14.5 s, 3 sentences |

- **Observations:**
  - **It works end to end**: spoken Devanagari came back, in sentence chunks.
  - **Cold start is huge** (8.5 s to first token): the server's first request with a new system prompt. `fryday-talk` now warms the LLM (and its prefix cache), Whisper and Kokoro before the first turn.
  - **ASR is the biggest warm cost** (about 2 s for 3 s of speech). Whisper encodes a padded 30 s window whatever the input length. To investigate in v4.
  - **First audio waits for the whole first sentence**: about 1.3 s between first token and first audio. Devanagari costs about 3.75 tokens per word (F3), so an 8-word sentence is about 30 tokens. The HLD's short first chunk (~6 words) is the v5 fix, and S0-8 weighs the script choice.
  - **Quality is weak.** ASR heard "कल सुबह **साथ बज़े** याद **देखे**" for "kal subah 7 baje yaad dilana", and the 4B model's Devanagari Hindi was clumsy and longer than the 1–3 short sentences asked for (10–15 s of audio).
  - Loading downloaded all 56 Kokoro voices (16 s); now only the configured voice is fetched.
- **Decision:**
  - Keep the skeleton; the warm turn (3.6 s stop → speech) is the baseline to beat. HLD target and the budget table come in F9.
  - v4: measure Whisper latency vs model size and input padding, and WER on the own-voice set.
  - v5: short first chunk, VAD endpointing, barge-in. Re-measure with live mic and playback.

## 2026-10-18: Scoring the Kokoro WAVs: predicted MOS vs ASR round trip vs ears (v3 learn 2)
- **Question:** can automatic metrics replace the owner's listening check, and which one catches the missing "rupaye"?
- **Setup:**
  - The 5 WAVs from the 2026-10-16 entry (Kokoro, 24 kHz). n=1 per file, one run.
  - **Naturalness:** UTMOS22 strong via SpeechMOS, torch.hub pinned to commit `ed25eac` (tag v1.2.0, MIT weights). Trained on English listening tests.
  - **Intelligibility:** mlx-whisper 0.4.3, `mlx-community/whisper-large-v3-turbo` revision `a4aaeec`, language forced. CER (jiwer 4.0.0) against the **intended normalised** text, with punctuation and spelling variants (ँ/ं, रुपए/रुपये) unified on both sides. A targeted check that the key words (सात, रुपये) come back as whole words.
  - Script: `spikes/learn/v3_tts_eval.py`.
- **Result:**

  | File | UTMOS | CER | Key words missing | Owner's ears |
  |---|---|---|---|---|
  | en | 4.50 | 10% | none | — |
  | hi_deva | 4.27 | 9% | सात (false alarm) | spot on |
  | hi_roman | 4.38 | 24% | सात (false alarm) | weak |
  | hi_deva_male | 4.11 | 11% | सात (false alarm) | — |
  | hi_digits | 4.31 | 24% | सात (false alarm), **रुपये** | best-sounding |

- **Observations:**
  - **UTMOS disagrees with the ears:** it ranks the mispronounced Roman file above Devanagari. All scores sit in 4.11–4.50, so with n=1 the gaps are noise, and Hindi is out of domain for the predictor.
  - **CER agrees with the ears:** Roman 24% vs Devanagari 9%. Whisper heard "काल सबसाथ बेजिम", the G2P mispronunciation.
  - **The round trip caught the missing रुपये** that UTMOS (4.31) and the owner's ears missed.
  - **ASR confound:** Whisper writes साथ for सात in every Hindi file. The two are near-homophones, and the owner heard सात correctly, so the key-word check raised a false alarm on all four. Whisper also gives बजी for बजे and merges "दो सौ पचास" into one word. English CER is all formatting ("friday", "7").
- **Decision:**
  - **TTS regression checks:** ASR round-trip CER plus entity checks (amounts, times, names) on a fixed sentence set, compared relative to a baseline, never against 0%.
  - **Hindi voice choice (v3 build):** decided by the owner's ears with A/B (CMOS-style) comparisons. UTMOS is only a "did quality crash?" alarm.
  - **First evidence for v4 (Whisper Hindi):** near-homophones, verb endings and number formatting are weak spots. Try an `initial_prompt` with "Fryday" in it, and measure properly in v4 (S0-8).

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

