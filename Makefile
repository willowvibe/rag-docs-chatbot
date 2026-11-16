# Works with either the Compose v2 plugin (`docker compose`) or the legacy
# standalone binary (`docker-compose`), whichever this machine has.
COMPOSE := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")
# Use the project venv when it exists; fall back to the system interpreter so CI,
# which has no .venv, still works.
PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

.PHONY: up down ingest serve test lint eval validate-golden verify-corpus

up:
	$(COMPOSE) up -d db

down:
	$(COMPOSE) down -v

ingest:
	$(PYTHON) -m ragchat.ingest.pipeline corpus/

serve:
	$(PYTHON) -m uvicorn ragchat.api:app --reload --port 8000

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

verify-corpus:
	sha256sum -c corpus/SHA256SUMS
