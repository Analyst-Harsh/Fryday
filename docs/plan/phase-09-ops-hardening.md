# Phase 9 — Ops hardening

| | |
|---|---|
| **Goal** | Make the Mac system operable: SLOs, alerts, backups, retention and scans. |
| **Depends on** | 8 |
| **Estimate** | 4–5 working days at 2–4 h/day |
| **HLD refs** | §7, §8 |
| **Status** | Not started |

## Scope
- Prometheus metrics, plus an SLO alert script that raises a desktop notification on an SLO breach or a stuck `unknown` approval.
- Nightly encrypted `pg_dump` (keep 7) and the first restore drill.
- 30-day retention, a `forget` command, and audio opt-in and delete.
- `pip-audit` and Trivy in CI; container images pinned by digest.
- `RUNBOOK.md` Mac-side entries: stuck approval, expired OAuth, Postgres restore, model rollback.

## Out of scope
- GPU heartbeat alerts and GPU runbook entries (phase 11).

## Exit criteria
- [ ] The alert fires on an injected SLO breach.
- [ ] The restore drill is done and documented.
- [ ] A test proves `forget` removes all of the user's data.
- [ ] The scans run in CI.

## What you learn
SRE basics: SLIs and SLOs, alerting, backup and restore, data retention.

## Units
_Written when this phase starts. Each unit is 2–3 h and follows the `superpowers:writing-plans` task format (files, failing test, implementation, verification, commit)._
