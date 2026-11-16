# Works with either the Compose v2 plugin (`docker compose`) or the legacy
# standalone binary (`docker-compose`), whichever this machine has.
COMPOSE := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

.PHONY: up down ingest serve test lint eval validate-golden verify-corpus

up:
	$(COMPOSE) up -d db

down:
	$(COMPOSE) down -v

ingest:
	python -m ragchat.ingest.pipeline corpus/

serve:
	uvicorn ragchat.api:app --reload --port 8000

test:
	pytest -q

lint:
	ruff check .

verify-corpus:
	sha256sum -c corpus/SHA256SUMS
