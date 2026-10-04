# Phase 8 — Voice turn manager

| | |
|---|---|
| **Goal** | Talk to Fryday end to end, including interrupting it. |
| **Depends on** | 6, 7 |
| **Estimate** | 6–8 working days at 2–4 h/day |
| **HLD refs** | §2, §8; architecture doc §9.1–9.2 |
| **Status** | Not started |

## Scope
- Browser microphone streaming PCM16 16 kHz over the WebSocket, with tap to start and tap to stop.
- Turn manager: epochs, barge-in, cancellation reaching the LLM and Triton.
- Ordered TTS playback through a jitter buffer, with backpressure.
- Spoken confirm phrase added to the approval card from phase 6.
- Admission control and a per-user daily minute cap.
- Golden Hinglish audio fixtures and `make e2e`.
- Chaos-lite: kill a backend mid-turn, and run the mock fault modes during voice turns.

## Out of scope
- Automatic end-of-speech detection (later).

## Exit criteria
- [ ] `make e2e` passes on the golden audio.
- [ ] A test proves barge-in drops stale chunks.
- [ ] Mac first-audio latency is measured and recorded.
- [ ] Admission control and the minute cap are tested.
- [ ] The system recovers in the chaos-lite tests.

## What you learn
Realtime streaming, backpressure, cancellation propagation.

## Units
_Written when this phase starts. Each unit is 2–3 h and follows the `superpowers:writing-plans` task format (files, failing test, implementation, verification, commit)._
