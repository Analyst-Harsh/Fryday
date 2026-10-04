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
- Timeouts for the stages that exist by then: LLM first token (2 s) and embed. Each ends in a polite failure message. Later phases add their own.
- The app binds to localhost only.

## Out of scope
- Memory, tools and voice.

## Exit criteria
- [ ] A text chat works end to end in the browser.
- [ ] A trace is visible in Langfuse.
- [ ] A scrubber test shows phone, address and name patterns absent from both the exported spans and the local logs.
- [ ] A migration test runs Alembic up from an empty DB.
- [ ] Tests cover the timeout path and WebSocket auth (bad token, bad origin, duplicate connection).

## What you learn
FastAPI WebSockets, LangGraph state and checkpointers, OpenTelemetry.

## Units
_Written when this phase starts. Each unit is 2–3 h and follows the `superpowers:writing-plans` task format (files, failing test, implementation, verification, commit)._
