.PHONY: sync test lint up down db-check

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
