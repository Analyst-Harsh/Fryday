# Phase 4 — Memory + non-spend tools

| | |
|---|---|
| **Goal** | Fryday remembers you and acts through safe tools. |
| **Depends on** | 3 |
| **Estimate** | 5–6 working days at 2–4 h/day |
| **HLD refs** | §1.2, §2 (step 5), §7 |
| **Status** | Not started |

## Scope
- Memories table with pgvector. Retrieve the top-k facts before each turn; extract new facts after each turn; a contradicting fact supersedes the old one and keeps its history.
- MCP tool client with Pydantic validation, one repair attempt, and tool output wrapped as untrusted data.
- Tools: notes, reminders, web search, and Google Calendar (OAuth, Fernet-encrypted tokens, `calendar.events.owned`).
- Non-spend retry: once, with the same idempotency key.

## Out of scope
- Spend tools (phase 6); voice.

## Exit criteria
- [ ] Retrieval and extraction have tests.
- [ ] Every tool has a contract test with recorded responses.
- [ ] An invalid tool call is repaired or politely refused.
- [ ] The live Calendar smoke test passes.

## What you learn
Embedding-based retrieval, MCP, tool-calling reliability, OAuth.

## Units
_Written when this phase starts. Each unit is 2–3 h and follows the `superpowers:writing-plans` task format (files, failing test, implementation, verification, commit)._
