# Phase 3 — Text agent skeleton

| | |
|---|---|
| **Goal** | First end-to-end slice: type Hinglish in the browser and get an answer from the local LLM, fully traced. |
| **Depends on** | 2 |
| **Estimate** | 6–7 working days at 2–4 h/day |
| **HLD refs** | §1.1–1.2, §7 (auth, privacy), §8 (timeouts, versioning) |
| **Status** | Not started |

## Scope
- FastAPI app with WebSocket protocol v1: a `v` field, the token sent in the first message, an Origin check, and newest-wins for duplicate connections.
- Postgres schema for users, sessions and turns, with Alembic migrations run on startup.
- LangGraph agent with a Postgres checkpointer.
- Thin React + Vite client with text chat.
- OpenTelemetry spans per stage, exported to Langfuse Cloud through the PII scrubber; the same scrubber covers local logs.
- Per-stage timeouts (HLD §8), each ending in a polite failure message.

## Out of scope
- Memory, tools and voice.

## Exit criteria
- [ ] A text chat works end to end in the browser.
- [ ] A trace is visible in Langfuse with PII scrubbed.
- [ ] Tests cover the timeout path and WebSocket auth (bad token, bad origin, duplicate connection).

## What you learn
FastAPI WebSockets, LangGraph state and checkpointers, OpenTelemetry.

## Units
_Written when this phase starts. Each unit is 2–3 h and follows the `superpowers:writing-plans` task format (files, failing test, implementation, verification, commit)._
