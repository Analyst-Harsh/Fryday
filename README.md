# Fryday

[![CI](https://github.com/Analyst-Harsh/Fryday/actions/workflows/ci.yml/badge.svg)](https://github.com/Analyst-Harsh/Fryday/actions/workflows/ci.yml)

Fryday is a Hinglish tap-to-talk voice assistant: it remembers you, uses tools (calendar, search, a mock grocery store with approvals), and speaks back. Its real purpose is **learning depth for AI engineering**: fine-tuning, audio, ONNX, TensorRT, Triton and CUDA principles, inside a production-grade system that runs on a 16 GB Mac, with a rented GPU only when the learning needs it.

- [High-level design](docs/specs/2026-10-04-fryday-hld-design.md)
- [Architecture overview](docs/specs/2026-10-04-fryday-architecture.md)
- [Implementation roadmap](docs/plan/00-roadmap.md)
- [Experiments log](docs/EXPERIMENTS.md) · [Runbook](docs/RUNBOOK.md) · [Security](SECURITY.md)

> **Status:** phase 0 (foundations): repo, tooling, CI and Postgres. No app logic yet.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (installs Python 3.12 for you)
- Docker with Compose v2 (Docker Desktop or OrbStack)
- [gitleaks](https://github.com/gitleaks/gitleaks) ≥ 8.19 (`brew install gitleaks`), used by the pre-commit hook

## Setup

```sh
make sync                   # uv sync --all-packages: venv + every workspace member
uv run lefthook install     # git hooks: once per clone (hooks aren't committed)
cp .env.example .env        # optional: compose has defaults; edit for real values
```

## Make targets

| Target | Does |
|---|---|
| `make sync` | Install all workspace members and dev tools |
| `make test` | Run pytest |
| `make lint` | ruff format check, ruff lint, pyright (strict) |
| `make up` | Start the `core` compose profile and wait until healthy |
| `make down` | Stop it (data volume kept; `docker compose --profile core down -v` wipes it) |
| `make db-check` | Print the installed pgvector version |
| `make llm` | Start the local model server (MLX, port 8080) |
| `make talk` | Voice chat in the terminal (push-to-talk; `FRYDAY_HANDS_FREE=1` for hands-free) |
| `make serve` | The app: WebSocket gateway + the built web UI on http://127.0.0.1:8000 (needs `FRYDAY_ACCESS_TOKEN` in `.env`) |
| `make web` | The React app's dev server on http://localhost:5173 (proxies `/ws` to `make serve`) |
| `make web-build` / `make web-test` | Build `web/dist` for `make serve` / run the frontend tests |

## Repo layout

| Path | What |
|---|---|
| `app/` | uv workspace member `fryday-app` (package `fryday`): the voice pipeline and the FastAPI gateway (`gateway.py`) |
| `web/` | The browser client: React + Vite + TypeScript |
| `infra/` | Files compose mounts (`infra/db/init.sql`); GPU scripts later |
| `compose.yaml` | Compose profiles; `core` = Postgres 16 + pgvector for now |
| `docs/` | Specs, roadmap and phase plans, experiments, runbook |

`ml/`, `eval/`, `client/`, `models/` and `spikes/` arrive in the phases that need them.

## Hooks and CI

- **pre-commit** (lefthook): ruff format and fix, pyright on staged files, gitleaks on staged changes.
- **pre-push**: the full format check, lint, type check and test suite.
- **CI** (GitHub Actions) on every PR and push to `main`: format check, lint, pyright and pytest; in parallel, the web job (tsc, vitest, vite build). Secret scanning (gitleaks) and the compose check run locally only: the pre-commit hook and `make up && make db-check`.

CI never runs real models.
