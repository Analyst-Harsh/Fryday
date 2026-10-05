.PHONY: sync test lint up down

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
