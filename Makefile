# Works with either the Compose v2 plugin (`docker compose`) or the legacy
# standalone binary (`docker-compose`), whichever this machine has.
COMPOSE := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")
# Use the project venv when it exists; fall back to the system interpreter so CI,
# which has no .venv, still works.
PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

.PHONY: up down ingest serve test lint install-dev eval validate-golden verify-corpus demo ask

up:
	$(COMPOSE) up -d db
	@for i in $$(seq 1 60); do \
	  $(COMPOSE) exec -T db pg_isready -U rag -d rag >/dev/null 2>&1 && break; \
	  [ $$i -eq 60 ] && echo "db never became ready; see: $(COMPOSE) logs db" && exit 1; \
	  sleep 1; \
	done

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

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

verify-corpus:
	sha256sum -c corpus/SHA256SUMS

validate-golden:
	$(PYTHON) -m ragchat.eval.validate_golden eval/golden.jsonl

eval: validate-golden
	$(PYTHON) -m ragchat.eval.run_eval

demo:
	$(COMPOSE) up -d --build
	@for i in $$(seq 1 60); do \
	  $(COMPOSE) exec -T db pg_isready -U rag -d rag >/dev/null 2>&1 && break; \
	  [ $$i -eq 60 ] && echo "db never became ready; see: $(COMPOSE) logs db" && exit 1; \
	  sleep 1; \
	done
	$(COMPOSE) exec -T app python -m ragchat.ingest.pipeline corpus/
	@echo "Open http://localhost:$$($(COMPOSE) port app 8000 | cut -d: -f2)"

ask:
	$(PYTHON) -c "import sys; from ragchat.config import load_settings; from ragchat.embeddings import Embedder; from ragchat.store import Store; from ragchat.answer import answer; q=sys.argv[1]; s=load_settings(); e=Embedder(); st=Store(s.database_url); r=answer(q, st.search(e.embed([q])[0], k=s.top_k), model=s.model); print(r.text); [print(f'  - {c.file} p.{c.page}') for c in r.citations]" "$(Q)"
