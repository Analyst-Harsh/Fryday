.PHONY: sync test lint up down db-check llm chat talk serve web web-build web-test

sync:
	uv sync --all-packages

test:
	uv run pytest -q

lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run pyright

up:
	docker compose --profile core up -d --wait

down:
	docker compose --profile core down

db-check:
	docker compose exec -T postgres sh -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB" -tAc "select extversion from pg_extension where extname='\''vector'\''"' | grep .

# Local LLM server (MLX, native on macOS: Docker has no Metal access). Ctrl-C stops it.
LLM_MODEL ?= mlx-community/Qwen3-4B-Instruct-2507-4bit
llm:
	uv run --with mlx-lm mlx_lm.server --model $(LLM_MODEL) --port 8080

# v0 terminal chat (needs `make llm` running in another terminal)
chat:
	uv run fryday-chat

# Push-to-talk voice chat: mic → Whisper → LLM → Kokoro → speaker (needs `make llm`; use headphones)
talk:
	uv run fryday-talk

# Browser voice chat: FastAPI gateway on :8000 (needs `make llm` and FRYDAY_ACCESS_TOKEN).
# Serves web/dist if built (`make web-build`); for development run `make web` alongside.
serve:
	uv run fryday-serve

# React app dev server on :5173, proxying /ws to `make serve`.
web:
	cd web && npm run dev

web-build:
	cd web && npm ci && npm run build

web-test:
	cd web && npm test
