# Phase 5 — Money path

| | |
|---|---|
| **Goal** | Safe ordering against a mock grocery server with fault injection. |
| **Depends on** | 4 |
| **Estimate** | 6–8 working days at 2–4 h/day |
| **HLD refs** | §7, §9; architecture doc §9.3 |
| **Status** | Not started |

## Scope
- Mock grocery MCP server as a separate process with its own order store (the source of truth), copying Swiggy's tool names, schemas and statuses.
- Fault modes, config-driven:
  - response dropped or delayed past 15 s after commit;
  - timeout before commit;
  - idempotency key `honoured` or `ignored` (the S0-10 flag);
  - `PENDING_PAYMENT` that ends `FAILED`;
  - eventually-consistent status lookup;
  - 5xx;
  - schema drift;
  - app killed between persisting the key and making the call.
- Approval state machine with compare-and-set and expiry by the DB clock; payload hash; read-back built from the stored row.
- Reconciler, run at startup and every 60 s.
- Audit log behind an INSERT-only DB role.
- Confirm card showing the amount, items and address as text. A **typed** confirm phrase plus a tap, both bound to `approval_id` + `payload_hash`. Phase 8 replaces the typed phrase with a spoken one.
- Approval timings: `pending` expires after 45 s and `approved` after 10 s, by the DB clock; a reconciler moves `executing` rows older than 30 s to `unknown`.
- LangGraph interrupt with a checkpointer and `thread_id`.
- MCP spend timeout of 15 s, then `unknown` (HLD §8).
- Prompt-injection test suite.

## Out of scope
- Real Swiggy (deferred); voice confirmation (phase 8 adds the spoken phrase on top of the card).

## Exit criteria
- [ ] Orders per approval ≤ 1 under every fault mode, checked by an automated test.
- [ ] Expired or mismatched confirmations are rejected.
- [ ] The audit role cannot UPDATE or DELETE.
- [ ] At least 10 injection fixtures, and the stored `payload_hash` is unchanged in every case.

## What you learn
State machines, idempotency, failure semantics (at-most-once, reconciliation), prompt-injection defence.

## Units
_Written when this phase starts. Each unit is 2–3 h and follows the `superpowers:writing-plans` task format (files, failing test, implementation, verification, commit)._
