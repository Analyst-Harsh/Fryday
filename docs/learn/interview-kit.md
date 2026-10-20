# Interview kit

This file is filled in at each version's interview session (I). It becomes the revision deck for v16.

For each version it holds:
- the questions asked, with your best answer written in your own words after the session;
- one STAR story, built from something real that happened in that version;
- the numbers to quote, taken from `EXPERIMENTS.md`.

## Template
```markdown
## vN: <title>
### Questions
1. **Q:** …
   **My answer:** …
### Design question
- **Q:** …  **Sketch:** …
### STAR story
- **Situation / Task / Action / Result:** …
### Numbers to quote
- …
```

---

## v3: Speech I, voice out (TTS)
### Questions
_Pending: the v3 interview questions haven't been asked yet._

### Design question
- **Q:** Design the TTS part of a voice assistant for 10,000 users. Replies must start speaking fast, sound natural, and read money and times correctly.
- **Sketch:**
  1. **Clarify:**
     - assume 10% of users are live at peak, so 1,000 sessions;
     - the assistant talks about 30% of the time, so about 300 TTS streams at once;
     - target first audio under 300 ms from text arriving;
     - Hindi/Hinglish plus English, over WebRTC/WebSocket.
  2. **Numbers:**
     - raw PCM is 300 × 384 kbit/s ≈ 115 Mbit/s;
     - Opus at 24 kbit/s brings that to about 7 Mbit/s, so encode;
     - GPUs = 300 / N, plus 30% headroom, where N = real-time streams per GPU from a load test (N = 50 gives about 8 GPUs).
  3. **Flow:**
     - client ⇄ gateway (auth, rate limit, session) → orchestrator;
     - the orchestrator streams LLM tokens to the chunker, then the normaliser, then a TTS request carrying text, voice and seq#;
     - the TTS service has a phrase cache, a priority queue, dynamic batching and a GPU pool (Triton);
     - PCM goes through the Opus encoder and back to the client in seq# order.
  4. **Decisions:**

     | Decision | Why |
     |---|---|
     | Stream at every stage: short first chunk (~6 words), then sentences | Time to first audio is what users feel; full sentences cost ~1.3 s in Fryday |
     | WFST rules plus structured tool data for amounts | Neural normalisation fails silently on numbers |
     | Dynamic batching with a 5–10 ms queue delay | Keeps GPUs busy without noticeable latency |
     | Priority queue, earliest deadline first | Deadline = now + buffered audio, so first chunks go first and nothing starves. Triton: `priority_levels` approximates it. |
     | Phrase cache keyed on (normalised text, voice, model version) | Fillers and greetings need no GPU |
     | Cancel on barge-in | Frees the GPU when the user interrupts |
     | Bounded queues | Backpressure: a slow client can't fill server memory |
     | Warm pools | Fryday cold start was 8.5 s vs 0.1 s warm |
     | Pinned model and voice, canary rollouts | A voice change is a product change |

  5. **Scaling and failures:**
     - autoscale on queue depth and p95 time to first audio, not CPU;
     - fallbacks: a smaller or CPU voice, then on-screen text;
     - timeouts on every request; `turn_id` + `seq#` for ordering.
  6. **Metrics:**
     - speed: time to first audio p50/p95, RTF, queue wait;
     - correctness: ASR round-trip WER plus a number/entity match on sampled audio;
     - quality: MOS/CMOS panels per release (UTMOS only as a rough, English-trained signal);
     - cost: GPU-hours per 1,000 minutes, cache hit rate.
  7. **Trade-offs:**
     - small fast model vs large natural model;
     - bigger batches are cheaper but give worse p95;
     - Devanagari input pronounces better but costs about 2.5× the LLM tokens of Roman.
  8. **What stands out:** time to first audio, barge-in cancellation, and normalising money. Most candidates talk only about the model.

### STAR story
_Pending._

### Numbers to quote
- Warm turn: 3.6 s from the user stopping to Fryday speaking (ASR ~2 s, first sentence ~1.3 s). Cold first token 8.5 s vs warm 0.1 s. Source: EXPERIMENTS 2026-10-19.
- Devanagari costs ~3.75 tokens per word vs ~1.5 for Roman Hinglish (EXPERIMENTS 2026-10-10).
- Kokoro-82M runs in fp32 at ~0.33 GB. Hindi G2P is espeak-ng: it drops ₹ and reads 7:30 as "सात तीस" (EXPERIMENTS 2026-10-20).
