# rag-docs-chatbot

Ask your documents, get cited answers — plus an eval score you can reproduce.

This repository is a document-QA app over a small, public-domain corpus. Answers are
generated from retrieved passages and carry citations back to the source document, and
the retrieval pipeline ships with an eval harness so the numbers can be regenerated from
a fresh clone.

**Status:** milestone 1 scaffold. The settings module and the corpus are in place;
ingestion, retrieval, the API and the eval arrive in later milestones.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Settings tests
.venv/bin/pytest tests/test_config.py -v

# Verify the bundled corpus has not changed
make verify-corpus
```

## Configuration

Every environment read lives in `ragchat/config.py`. Copy `.env.example` to `.env` and
fill in what you need; the app runs without an Anthropic key for tests and offline work.
Rate limiting (the demo lane) is off by default and, when enabled, requires the Upstash
REST URL.

| Variable | Default |
|---|---|
| `DATABASE_URL` | `postgresql://rag:rag@localhost:5433/rag` |
| `ANTHROPIC_API_KEY` | empty |
| `RAG_MODEL` | `claude-sonnet-5-5` |
| `RAG_CHEAP_MODEL` | `claude-haiku-5-5` |
| `RAG_TOP_K` | `5` |
| `MAX_QUESTION_CHARS` | `500` |
| `RATE_LIMIT_ENABLED` | `0` |
| `PER_IP_PER_HOUR` | `20` |
| `DAILY_CAP` | `500` |
| `UPSTASH_REDIS_REST_URL` | empty |
| `UPSTASH_REDIS_REST_TOKEN` | empty |

## Corpus

`corpus/` holds two public-domain NIST Special Publications, included unmodified so the
eval is reproducible from a fresh clone:

- `nist-sp-800-63b.pdf` — NIST SP 800-63B-4, *Digital Identity Guidelines: Authentication
  and Lifecycle Management*
- `nist-sp-800-63-4.pdf` — NIST SP 800-63-4, *Digital Identity Guidelines*

Licensing is documented in `corpus/LICENSE.corpus.md`; the files' SHA-256 hashes are
recorded in `corpus/SHA256SUMS` and checked by `make verify-corpus`.

## Layout

```
ragchat/       application package (settings live in config.py)
corpus/        public-domain source documents + provenance
tests/         pytest suite
```

## Licence

MIT — see `LICENSE`. The corpus documents are U.S. Government works in the public domain
(see `corpus/LICENSE.corpus.md`).
