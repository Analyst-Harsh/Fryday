# Runbook

What to do when something breaks (HLD §8). Each entry: **symptom → check → fix → verify**.

## Browser says "Getting ready" or keeps reconnecting
- **Check:** `curl -s localhost:8000/readyz`. `warming up` = Whisper and Kokoro are still loading (~10 s); `model server unreachable` = `make llm` isn't running or is still loading its model.
- **Fix:** start or wait for `make llm`; the page reconnects by itself (close 1013, then retry).
- **Verify:** `/readyz` returns `{"ready": true}` and the page says "Listening".

## Browser says "That token was refused"
- **Check:** `FRYDAY_ACCESS_TOKEN` in `.env` is what was pasted (the server closed with 4401).
- **Fix:** paste the token again; to rotate it, change `.env` and restart `make serve`.
- **Verify:** the page reaches "Listening".

## Postgres won't start or is unhealthy
- **Check:** `docker compose ps`; `docker compose --profile core logs postgres`.
- **Fix:** read the error. If `init.sql` changed or the volume is broken, run `docker compose --profile core down -v`. This **deletes all data**; restore from backup (below).
- **Verify:** `make up && make db-check`.

## GPU box dead or tunnel down
_Filled in phase 11._

## Stuck `unknown` approval
_Filled in phase 9._

## Expired OAuth token
_Filled in phase 9._

## Postgres restore
_Filled in phase 9 (nightly `pg_dump`, keep 7, quarterly restore drill)._

## Model rollback
_Filled in phase 9._
