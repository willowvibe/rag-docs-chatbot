# rag-docs-chatbot

Ask your documents, get cited answers — plus an eval score you can reproduce.

This repository is a document-QA app over a small, public-domain corpus. Answers are
generated from retrieved passages and carry citations back to the source document, and
the retrieval pipeline ships with an eval harness so the numbers can be regenerated from
a fresh clone.

**Status:** pre-release. What works today: PDF, Markdown and HTML ingestion; local BGE
embeddings; pgvector storage and cosine search; LLM answers via Ollama Cloud (configurable
model, default `gpt-oss:120b`) grounded in retrieved passages with `[n]` citations and an
explicit refusal; a rate-limited FastAPI service and
web UI; a 30-question eval harness; a Docker quickstart; and CI. The published eval
report, a demo video and a live demo are coming.

## Quick start

Requires Docker and Python 3.12. Answering needs an `OLLAMA_API_KEY` (Ollama Cloud), or an `ANTHROPIC_API_KEY` with
`LLM_PROVIDER=anthropic`; ingestion,
retrieval and the tests work without one.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

make up       # start Postgres with pgvector
make ingest   # prints: total=386 chunks across 2 documents
make serve    # http://localhost:8000

# In another shell:
make test
```

See [docs/architecture.md](docs/architecture.md) for the ingest and query paths.

## Configuration

Every environment read lives in `ragchat/config.py`. Copy `.env.example` to `.env` and
fill in what you need; the app runs without an LLM key for tests and offline work.
Rate limiting (the demo lane) is off by default and, when enabled, requires the Upstash
REST URL.

If another Postgres already holds port 5433, bring the database up on a different host port
with `RAG_DB_PORT=5435 make up` and point `DATABASE_URL` at the same port.

| Variable | Default |
|---|---|
| `DATABASE_URL` | `postgresql://rag:rag@localhost:5433/rag` |
| `RAG_DB_PORT` | `5433` |
| `LLM_PROVIDER` | `ollama` (or `anthropic`) |
| `OLLAMA_API_KEY` | empty |
| `OLLAMA_HOST` | `https://ollama.com` |
| `ANTHROPIC_API_KEY` | empty (only with `LLM_PROVIDER=anthropic`) |
| `RAG_MODEL` | `gpt-oss:120b` (anthropic: `claude-sonnet-5-5`) |
| `RAG_CHEAP_MODEL` | `gpt-oss:20b` (anthropic: `claude-haiku-5-5`) |
| `RAG_TOP_K` | `5` |
| `MAX_QUESTION_CHARS` | `500` |
| `RATE_LIMIT_ENABLED` | `0` |
| `PER_IP_PER_HOUR` | `20` |
| `DAILY_CAP` | `500` |
| `UPSTASH_REDIS_REST_URL` | empty |
| `UPSTASH_REDIS_REST_TOKEN` | empty |

## Deploy

The app runs in two lanes. **Self-host** means running it locally or in Docker with your
own `OLLAMA_API_KEY` (or Anthropic key). **Hosted demo** means running it on Vercel, where the owner's key
lives in the project's environment variables — never in the browser — and per-IP and
per-day limits backstop it. `RATE_LIMIT_ENABLED` ships as `0`; only the demo deployment
sets it to `1`.

Prerequisites:

- A Postgres database with the `vector` extension (the Neon free tier works).
- An Upstash Redis database, for the REST URL and token that hold the limit counters.
- This repository imported as a Vercel project.

Ingest against the hosted database **before** the first request: point `DATABASE_URL` at
it and run `make ingest`. The deployed app only reads the `chunks` table; `corpus/` and
`eval/` are excluded from the upload by `.vercelignore`, so there is nothing to ingest
inside the deployment.

Set these in the Vercel project settings:

- `OLLAMA_API_KEY` (or `LLM_PROVIDER=anthropic` plus `ANTHROPIC_API_KEY`)
- `DATABASE_URL`
- `UPSTASH_REDIS_REST_URL`
- `UPSTASH_REDIS_REST_TOKEN`
- `RATE_LIMIT_ENABLED=1` (demo only; the shipped default is `0`)
- `PER_IP_PER_HOUR=20`
- `DAILY_CAP=500`
- `FASTEMBED_CACHE_PATH=/tmp/fastembed`

The first four carry deployment-specific credentials; keep them in the project settings
and out of the repository. `FASTEMBED_CACHE_PATH` is set explicitly because the function
filesystem is read-only apart from `/tmp`, so the model cache must live there. fastembed's
own fallback is `/tmp/fastembed_cache`, which makes this an explicit choice rather than a
fix for a crash.

Cold starts are slow: importing the app loads the tiktoken `cl100k_base` encoding (a small
download when the cache is cold), and the first embed downloads the BGE model into
`/tmp`. Subsequent requests on a warm instance are fast.

Vercel's FastAPI framework preset serves the app at its own paths — `/`, `/api/*` and
`/static/*` — with no rewrites. `vercel.json` only sets `maxDuration` to 300 s, the
Fluid-compute default and the maximum on the Hobby plan, so it pins the default rather
than extending it; that leaves room for a cold start. If you ever want the `/api`
file-function lane instead of the preset, set the project's Framework Preset to "Other".
Vercel runs the Python version named in `.python-version` (3.12); the `Dockerfile` and CI
use 3.12 too, so every environment runs the same interpreter.

If the Python bundle exceeds Vercel's size limit, the same `Dockerfile` runs unchanged on
Render or Fly — which is why the app contains no Vercel-specific code.

To confirm the limits actually bite, run this against a deployment with
`RATE_LIMIT_ENABLED=1`. Run it **once, on a fresh hour, from a single machine**: the
per-IP counter is per hour, so a second run in the same hour begins with `429`s. Vercel
overwrites a client-supplied `X-Forwarded-For` and sets `x-real-ip` to the same value, so
the `x-forwarded-for` header below is inert on Vercel; the loop works because all 21
requests share one public IP. The header is kept because a non-Vercel proxy would honour
it.

```bash
URL=https://<deployment>.vercel.app
for i in $(seq 1 21); do
  curl -s -o /dev/null -w "%{http_code}\n" -X POST "$URL/api/chat" \
    -H 'content-type: application/json' -H 'x-forwarded-for: 203.0.113.7' \
    -d '{"question":"What is the minimum password length?"}'
done | sort | uniq -c
```

Twenty requests should print `200` and the twenty-first `429`. Check that the UI shows
the 429 message and still renders, and that the key is absent from every client-side file.
This prints `0` for each of `/`, `/static/app.js` and `/static/styles.css`:

```bash
URL=https://<deployment>.vercel.app
for p in / /static/app.js /static/styles.css; do
  printf '%s ' "$p"; curl -s "$URL$p" | grep -c sk-ant
done
```

## Corpus

`corpus/` holds two public-domain NIST Special Publications, included unmodified so the
eval is reproducible from a fresh clone:

- `nist-sp-800-63b.pdf` — NIST SP 800-63B-4, *Digital Identity Guidelines: Authentication
  and Lifecycle Management*
- `nist-sp-800-63-4.pdf` — NIST SP 800-63-4, *Digital Identity Guidelines*

Licensing is documented in `corpus/LICENSE.corpus.txt`; the files' SHA-256 hashes are
recorded in `corpus/SHA256SUMS` and checked by `make verify-corpus`.

## Layout

```
ragchat/       application package (settings live in config.py)
ragchat/eval/  golden-set validator and eval runner
api/           Vercel entry point
web/           static chat UI
corpus/        public-domain source documents + provenance
eval/          golden questions and generated report
docs/          architecture notes
tests/         pytest suite
```

## Licence

MIT — see `LICENSE`. The corpus documents are U.S. Government works in the public domain
(see `corpus/LICENSE.corpus.txt`).
