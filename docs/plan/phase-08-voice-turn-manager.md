# Phase 8 — Voice turn manager

| | |
|---|---|
| **Goal** | Talk to Fryday end to end, including interrupting it. |
| **Depends on** | 5, 7 |
| **Estimate** | 6–8 working days at 2–4 h/day |
| **HLD refs** | §2, §8; architecture doc §9.1–9.2 |
| **Status** | Not started |

## Scope
- Browser microphone streaming PCM16 16 kHz over the WebSocket, with tap to start and tap to stop.
- Turn manager: epochs, barge-in, cancellation reaching the LLM and Triton.
- Ordered TTS playback through a jitter buffer, with backpressure.
- The spoken confirm phrase replaces the typed one. A low-confidence ASR result is rejected as a confirmation (HLD §7).
- A filler line is spoken between tool-call validation and the MCP result.
- TTS chunk timeout of 2 s (HLD §8).
- OTel spans `asr` and `tts_chunk` are added to the turn trace.
- Admission control and a per-user daily minute cap.
- Golden Hinglish audio fixtures: own voice plus TTS-generated, kept apart from all training data. `make e2e` runs them.
- Chaos-lite: kill a backend mid-turn, and run the mock fault modes during voice turns.

## Out of scope
- Automatic end-of-speech detection (later).

## Exit criteria
- [ ] `make e2e` passes on the golden audio.
- [ ] A test proves barge-in drops stale chunks.
- [ ] Mac p95 first audio over 20 golden turns is recorded against the ≤ 4 s SLO.
- [ ] Admission control and the minute cap are tested.
- [ ] The system recovers in the chaos-lite tests.

## What you learn
Realtime streaming, backpressure, cancellation propagation.

## Units
_Written when this phase starts. Each unit is 2–3 h and follows the `superpowers:writing-plans` task format (files, failing test, implementation, verification, commit)._
