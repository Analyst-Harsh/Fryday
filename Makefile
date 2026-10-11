.PHONY: sync test lint up down db-check llm chat

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
