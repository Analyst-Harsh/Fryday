# Runbook

What to do when something breaks (HLD §8). Each entry: **symptom → check → fix → verify**.

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
