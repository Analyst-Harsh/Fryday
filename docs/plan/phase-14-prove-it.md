# Phase 14 — GPU sprint 3: prove it

| | |
|---|---|
| **Goal** | Show the numbers: load, cost and resilience, plus a demo. |
| **Depends on** | 13 |
| **Estimate** | 4–6 working days at 2–4 h/day |
| **HLD refs** | §2, §8, §11 (M6) |
| **Status** | Not started |

## Scope
- k6 or Locust WebSocket load script, written and dry-run on the Mac first.
- E9: KV cache and continuous batching vs concurrency.
- E10: maximum concurrency before the SLO breaks.
- E11: cost per conversation-minute.
- Set the admission limit from the data; measure p95 first audio against the SLO.
- Chaos: drop the SSH tunnel mid-spend; the reconciler must resolve it.
- Record the demo; write up README and portfolio notes.

## Out of scope
- Phase 2 features (vision, wake word, and so on).

## Exit criteria
- [ ] E9–E11 are written up.
- [ ] The admission limit is set.
- [ ] GPU p95 is measured against the SLO.
- [ ] The tunnel-drop chaos test passes.
- [ ] The demo is recorded.

## What you learn
Load testing, capacity planning, cost modelling.

## Units
_Written when this phase starts. Each unit is 2–3 h and follows the `superpowers:writing-plans` task format (files, failing test, implementation, verification, commit)._
