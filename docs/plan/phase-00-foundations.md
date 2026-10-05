# Phase 0 — Foundations

| | |
|---|---|
| **Goal** | A clean public repo where every later phase can add code with tests and CI from day one. |
| **Depends on** | — |
| **Estimate** | 2–3 working days at 2–4 h/day |
| **HLD refs** | §1.7, §10 |
| **Status** | In progress |

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

### Unit 1: publish safely, uv workspace, lint/type/test, Makefile (~3 h)
1. **First:** paste these units into the **Units** section of `phase-00-foundations.md`, as the roadmap says, and set Status → In progress.
2. **Prerequisites:** `brew install gitleaks` (needs ≥ 8.19 for the `gitleaks git` syntax).
3. **Pre-publish checks:**
   - `gitleaks git -v` on the full history must be clean. The reviewer's grep found nothing.
   - Skim `docs/voice_assistant` (the original draft) for personal content.
4. Add `.gitignore` and `.gitattributes`:
   - `.gitignore`: Triage Bot's, plus `.venv/`, `.env.*` with a `!.env.example` exception, and `leak.txt` as a guard for unit 2.
   - `.gitattributes`: `uv.lock linguist-generated=true`.
   - Commit together with the step-1 phase-file edit: `chore: gitignore, gitattributes, phase-0 units`.
5. Create the repo: `gh repo create Analyst-Harsh/Fryday --public --source . --push`. Then make the branch `phase-0-foundations`. All further work lands through a PR, so CI runs on it.
6. Root `pyproject.toml`:
   - `[tool.uv.workspace] members = ["app"]`
   - `[dependency-groups] dev = ["ruff", "pyright", "pytest", "lefthook"]`
   - ruff: copy Triage Bot's `[tool.ruff*]` blocks with these changes:
     - `target-version = "py312"`;
     - per-file ignore `"**/tests/**/*.py" = ["S101","S105","S106"]` (Triage Bot's `tests/**` doesn't match `app/tests/`);
     - `extend-exclude = ["spikes"]`.
   - pyright: strict, `pythonVersion = "3.12"`, `include = ["app"]`, `exclude` adds `spikes`, `venvPath = "."`, `venv = ".venv"`, `reportMissingTypeStubs = false`.
   - pytest: `testpaths = ["app/tests"]`, `addopts = "--import-mode=importlib"`. Don't set `pythonpath`.
7. `app/pyproject.toml`:
   - `name = "fryday-app"`, `version = "0.0.1"`, `requires-python = ">=3.12,<3.13"`;
   - `[build-system] requires = ["uv_build>=0.11,<0.12"]`, `build-backend = "uv_build"`;
   - `[tool.uv.build-backend] module-name = "fryday"`. Without this, uv_build looks for a `fryday_app` module.
8. **Failing test first:**
   - Run `uv sync --all-packages` first. In a virtual workspace root, plain `uv run` doesn't install the `app` member, so the test would fail with an import error instead of the assertion.
   - `app/tests/test_smoke.py` asserts `fryday.__version__ == "0.0.1"`. Run `uv run pytest`; it must fail on that assertion.
9. Implement `app/src/fryday/__init__.py` with `__version__ = "0.0.1"`.
10. `Makefile` (all `.PHONY`):
    - `sync`: `uv sync --all-packages`. The flag guarantees the `app` member installs from the virtual root.
    - `test`: `uv run pytest -q`
    - `lint`: `uv run ruff format --check . && uv run ruff check . && uv run pyright`
    - `up`: `docker compose --profile core up -d --wait`
    - `down`: `docker compose --profile core down`
11. **Verify:** `make sync && uv run python -c "import fryday"`, then `make test` and `make lint` both pass. Commit `chore: uv workspace, ruff, pyright, pytest, Makefile`.

### Unit 2: hooks, gitleaks and the secrets policy (~2 h)
1. `lefthook.yml`, adapted from Triage Bot:
   - `pre-commit` (sequential):
     - ruff format, with `stage_fixed`;
     - `ruff check --fix`, with `stage_fixed`;
     - pyright on staged `*.py`;
     - gitleaks: `gitleaks git --pre-commit --staged --redact --verbose`.
   - `pre-push` (parallel): format check, ruff check, pyright, pytest.
   - Note: GUI git clients may lack `/opt/homebrew/bin` on PATH. Commit from the terminal.
2. **After** `lefthook.yml` exists, run `uv run lefthook install` and check that `.git/hooks/pre-commit` exists. If you install first, lefthook writes a default config and no pre-commit hook.
3. `.env.example`: `POSTGRES_USER=fryday`, `POSTGRES_PASSWORD=change-me`, `POSTGRES_DB=fryday`, `POSTGRES_PORT=5432`, plus a comment that later phases add keys here.
4. `SECURITY.md`: a short disclosure section, adapted from Triage Bot, plus a **Secrets policy**:
   - secrets only in an untracked `.env`;
   - `.env.example` holds placeholders only;
   - gitleaks runs in pre-commit and CI;
   - if a secret leaks, rotate it first, then purge it from git history.
5. **Verify (exit criterion 3), with guard rails:**
   - Confirm `.git/hooks/pre-commit` exists, so the commit actually goes through the hook.
   - Write `ghp_` + 36 *random* alphanumerics (not a repeated pattern) into `leak.txt`.
   - `git add -f leak.txt && git commit -m test`. This must be **blocked** with a redacted finding.
   - Clean up with `git reset HEAD leak.txt && rm leak.txt`. Never use `--no-verify` here.
   - Guard: if `git log -1 --stat` shows `leak.txt`, the commit went through. Run `git reset HEAD~1 && rm leak.txt` before doing anything else, and never push it. A mixed reset keeps the unit's other uncommitted work; `--hard` would throw it away.
   - Paste only the redacted output into the PR description.
6. Commit `chore: lefthook hooks, gitleaks, secrets policy`.

### Unit 3: docker compose `core` profile (~2 h)
1. `compose.yaml`:
   - A top comment: per HLD §6, `core` = app + Postgres + model backends, and later phases add to this profile.
   - Service `postgres`, `profiles: ["core"]`.
   - Image `pgvector/pgvector:pg16@sha256:<digest>`, with the digest from `docker buildx imagetools inspect pgvector/pgvector:pg16`.
   - `environment:` set to `POSTGRES_USER: ${POSTGRES_USER:-fryday}`, `POSTGRES_DB: ${POSTGRES_DB:-fryday}` and `POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-change-me}`. Compose reads `.env` itself for this interpolation, so no `env_file` is needed, and these defaults let a fresh clone start.
   - Port `127.0.0.1:${POSTGRES_PORT:-5432}:5432` (localhost only, HLD §7).
   - Named volume `pgdata`.
   - Volume `./infra/db/init.sql:/docker-entrypoint-initdb.d/init.sql:ro`. This mounts the file, not over the directory.
   - Healthcheck `["CMD-SHELL", "pg_isready -h 127.0.0.1 -U $${POSTGRES_USER} -d $${POSTGRES_DB}"]`, interval 5s, retries 10. The **TCP** check only passes after the init scripts have run. A socket check can pass early, which races `--wait`.
2. `infra/db/init.sql`: `CREATE EXTENSION IF NOT EXISTS vector;`. The pgvector image does not create the extension itself.
3. Add a Makefile target `db-check`: `docker compose exec -T postgres sh -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB" -tAc "select extversion from pg_extension where extname='\''vector'\''"' | grep .`. CI uses the same target.
4. **Verify (exit criterion 2):**
   - `make up` works on a fresh clone with no `.env`; `docker compose ps` shows `healthy`.
   - `make db-check` prints a version.
   - `make down`.
   - If Triage Bot's DB is using port 5432, set `POSTGRES_PORT=5433` in `.env`.
5. Add the one-line architecture §7 note about where compose lives. Commit `feat(infra): compose core profile with Postgres 16 + pgvector`.

### Unit 4: CI, README, doc stubs, PR (~2.5 h)
1. `.github/workflows/ci.yml`, shaped like `bi_frost.yml`:
   - Triggers: push and PR to `main`, plus `workflow_dispatch`.
   - `concurrency` with cancel-in-progress; workflow `permissions: contents: read`.
   - Pin every action by **commit SHA**, with a version comment (look the SHA up with `gh api repos/<o>/<r>/git/ref/tags/<tag>`): `actions/checkout` v7.0.1, `astral-sh/setup-uv` v10.2.0, `gitleaks/gitleaks-action` v3.x.
   - Job `checks`:
     - checkout;
     - setup-uv with cache on `uv.lock`;
     - `uv sync --locked --all-packages`;
     - format check → `ruff check` → `pyright` → `pytest`, each with an `id`;
     - an `if: always()` job-summary table, same as Triage Bot.
   - Job `secrets`:
     - checkout with `fetch-depth: 0`;
     - gitleaks-action with `GITHUB_TOKEN`;
     - job permissions `contents: read`, `pull-requests: read`.
   - Job `compose`: `make up`, then `make db-check`, then `make down` with `if: always()`.
   - `timeout-minutes` on every job.
2. `README.md`:
   - one-paragraph pitch, linking to the HLD, architecture and roadmap;
   - prerequisites: uv, Docker, gitleaks ≥ 8.19;
   - setup: `make sync && uv run lefthook install && cp .env.example .env`;
   - the make targets;
   - a CI badge.
3. Stubs:
   - `docs/EXPERIMENTS.md`: a template entry with date, question, setup, result, decision.
   - `docs/RUNBOOK.md`: headings from HLD §8, each marked "filled in phase 9/11".
4. Open a PR with `gh pr create`. **Verify (exit criterion 1):** all 3 CI jobs are green.
5. Final commit on the branch: in the phase file, set Status → Done and tick the exit criteria. Squash-merge, then confirm CI is green on `main` (`gh run list --branch main`).
