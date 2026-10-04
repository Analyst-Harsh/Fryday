# Phase 0 — Foundations

| | |
|---|---|
| **Goal** | A clean public repo where every later phase can add code with tests and CI from day one. |
| **Depends on** | — |
| **Estimate** | 2–3 working days at 2–4 h/day |
| **HLD refs** | §1.7, §10 |
| **Status** | Not started |

## Scope
- Create the public GitHub repo and push the existing docs.
- Set up a `uv` workspace with Python 3.12, plus ruff, pyright and pytest, and a `Makefile` (`test`, `lint`, `up`, `down`).
- Add a pre-commit config: ruff, plus gitleaks for secrets.
- Add GitHub Actions CI running lint, type-check and tests.
- Add docker compose with a `core` profile: Postgres 16 + pgvector.
- Add `.env.example` and a secrets policy (no secrets in git).
- Add a README and stubs for `EXPERIMENTS.md` and `RUNBOOK.md`.

## Out of scope
- Any app logic, client or models.

## Exit criteria
- [ ] `make test` and CI are green on a first trivial test.
- [ ] `docker compose --profile core up` gives a healthy Postgres with the `vector` extension.
- [ ] gitleaks blocks a planted fake secret.

## What you learn
uv workspaces, CI pipelines, compose profiles, pre-commit hooks.

## Units
_Written when this phase starts. Each unit is 2–3 h and follows the `superpowers:writing-plans` task format (files, failing test, implementation, verification, commit)._
