.PHONY: up down ingest serve test lint eval validate-golden verify-corpus

up:
	docker compose up -d

down:
	docker compose down -v

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
