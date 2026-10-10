# Learning journal

This is an append-only log, newest entry first. Write one entry per session (it takes about 5 minutes).

## Mental map
Redraw this weekly in your own words. Each box is filled in by the versions listed under it.

```mermaid
flowchart LR
    C["Client<br/>mic, playback, approval card<br/><i>v4 CLI, v5 browser</i>"] -->|WebSocket| G["App: gateway + turn manager<br/><i>v5</i>"]
    G --> ASR["ASR<br/><i>v4, v8 Triton, v13 LoRA, v14 TRT</i>"]
    G --> A["Agent<br/>tools, approvals<br/><i>v1, v6</i>"]
    A --> LLM["LLM server<br/>OpenAI API<br/><i>v0 MLX, v9 vLLM, v10 quant, v11/v13 LoRA</i>"]
    A --> M["Memory<br/>embed + pgvector<br/><i>v2, v8, v14 TRT</i>"]
    A --> T["Tools / MCP<br/><i>v1, v6</i>"]
    G --> TTS["TTS + normaliser<br/><i>v3, v8</i>"]
    M --> DB[("Postgres<br/>memories, approvals, audit<br/><i>v1, v2, v6, v7b</i>")]
    A --> DB
    E["Eval + observability<br/><i>v7a, v7b</i>"] -.-> LLM
    E -.-> ASR
```

## Still don't get (re-ask after 1 week and again after 4 weeks)
| Added | Question | Re-asked (1 week) | Re-asked (4 weeks) |
|---|---|---|---|
| 2026-10-08 | What is a model **contract** (app ↔ model-server API shape, so backends can be swapped), and how is it different from **structured output** (the model's text following a schema)? | due 2026-10-15 | due 2026-11-05 |
| 2026-10-09 | (F2 quiz, skipped; ask at the F checkpoint) What are query, key and value? Why does a low temperature make output predictable? Why does an LLM need attention instead of fixed word embeddings? | due F session 10 | — |

## Entry template
```markdown
### YYYY-MM-DD: <version> session N (learn | build | interview)
- **Built or watched:**
- **Predicted vs actual:**
- **Surprised me:**
- **I can now explain:**
- **Still don't get:**
- **Tomorrow's first step:**
```

---

### 2026-10-10: v0 learn 1 + build 1: logits, decoding, first Fryday chat
- **Built or watched:**
  - `v0_logits.py`: real Qwen logits, shape (1, 38, 151936).
  - Temperature, top-k and top-p on real numbers.
  - Built `fryday-chat`: config, the `LLMClient` seam, JSON logs, history trimming, error handling.
- **Predicted vs actual:** —
- **Surprised me:**
  - At temperature 1.5, top-p 0.9 keeps **894** tokens (vs 4 at 0.3).
  - The first live chat hit a repetition loop (484 chunks), and I couldn't reproduce it in 18 runs.
  - The model mixes up "mera" (whose name?): it answered "Mera naam Fryday".
- **I can now explain:**
  - Logits vs probabilities (softmax only cares about differences).
  - Greedy, temperature, top-k, top-p, beam search, logprobs.
  - Why temperature 0 can still vary (near-ties plus floating point).
  - Low temperature for tool calls.
  - Why `max_retries=0` for streaming.
  - Why history keeps the system prompt first (prefix cache).
- **Still don't get:** —
- **Next:** v0 build 2, a contract test against a stub server plus OpenTelemetry spans.

### 2026-10-10: F session 7 (learn): what model hosting means
- **Built or watched:**
  - Hosting layers: weights, runtime, server, contract, app.
  - Weight formats.
  - Memory math (params × bytes).
  - Ran Qwen3-4B 4-bit with `mlx_lm.generate` and with `mlx_lm.server` plus `f7_probe.py`.
  - Also: a PyTorch primer (`f4_pytorch_primer.py`), plus walkthroughs of (B, T, C), nn.Linear and one attention head by hand.
- **Predicted vs actual:**

  | | Predicted | Actual |
  |---|---|---|
  | TTFT | 1–2 s | 0.53 s cold, 0.05–0.11 s warm |
  | Speed | 10 tok/s | about 42–55 tok/s |
  | RAM | about 4 GB | about 2.4–2.6 GB, which matches 4B × 0.5 bytes plus scales |

- **Surprised me:**
  - Speed was 4–5× faster than I guessed.
  - The prompt cache made a repeated prompt's TTFT about 10× smaller.
  - The model claimed it had set a reminder when it had no tool.
- **I can now explain:**
  - What each hosting layer does.
  - Why the contract lets the Mac (MLX) and the GPU (vLLM) swap.
  - Why 4-bit is about 2 GB.
  - Why warm-up exists.
- **Still don't get:** —
- **Next:** short F checkpoint, then v0 (LLM hosting I: build).

### 2026-10-10: F session 3 (learn): tokenisation
- **Built or watched:** Karpathy, *Let's build the GPT Tokenizer*. Toy: `spikes/learn/f3_tokens.py` on Qwen's real tokenizer.
- **Predicted vs actual:** I predicted English < Roman Hinglish < Devanagari ✅. The actual counts were 8 / 12 / 30 tokens.
- **Surprised me:**
  - Devanagari costs 3.75× English.
  - Byte-level BPE splits single Devanagari characters across tokens (the `�` pieces).
- **I can now explain:**
  - BPE sits between characters and words.
  - Byte-level BPE covers every language, at a cost for non-English scripts.
  - More tokens means more latency and more KV-cache memory.
- **Still don't get:** —
- **Tomorrow's first step:** F session 4, building a tiny GPT from scratch (part 1/2).

### 2026-10-09: F session 2 (learn): neural nets → transformers
- **Built or watched:** 3Blue1Brown neural nets, *Transformers* and *Attention*. Toy: `spikes/learn/f2_attention.py`, one attention head by hand.
- **Predicted vs actual:** skipped.
- **Surprised me:**
  - "bank" moves toward water or money purely from its neighbours.
  - The weights come out flat without the learned W_q and W_k.
- **I can now explain:**
  - Attention as three steps: score (q·k), normalise (softmax), blend (V).
  - Temperature sharpens or flattens the next-token distribution.
  - Past tokens' K and V never change, so they're cached (the KV cache).
- **Still don't get:** quiz deferred (see the table).
- **Tomorrow's first step:** F session 3, Karpathy's tokenizer lecture, plus a Hinglish token-count toy.

### 2026-10-08: F session 1 (learn): the big picture
- **Built or watched:** Karpathy, *Intro to Large Language Models*. Walked through one Fryday turn (architecture doc §1 and §4).
- **Predicted vs actual:** I predicted "model processing" eats most of the ~2 s. More precisely, ASR is the biggest cost on the Mac (1–2 s), and the LLM's time to its first chunk matters most on the GPU. Streaming hides the rest.
- **Surprised me:**
  - The app is a **hub**, not a pipeline: models never call each other.
  - The LLM can be called twice in one turn (once to decide on a tool, once to phrase the reply).
- **I can now explain:**
  - An LLM is two files, weights plus a run program, and the hosting layer (MLX, vLLM, Triton) is that run program.
  - The LLM only *proposes* tool calls. The app validates them and executes them.
  - Pretraining gives the model knowledge. Fine-tuning gives it behaviour and format.
  - Streaming overlaps LLM and TTS.
- **Still don't get:** contract vs structured output (I mixed them up in the quiz; see the table above).
- **Tomorrow's first step:** F session 2, the 3Blue1Brown neural-network refresher, then *Transformers* and *Attention*.
