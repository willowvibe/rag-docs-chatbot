"""One FastAPI app: JSON API plus the static chat page."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ragchat.answer import answer as answer_fn
from ragchat.config import load_settings
from ragchat.embeddings import Embedder
from ragchat.ratelimit import check_and_consume
from ragchat.store import Store

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="rag-docs-chatbot", docs_url="/api/docs", openapi_url="/api/openapi.json")

# The caches are module-private; the accessor functions are what the tests patch, so
# `monkeypatch.setattr(api, "_store", ...)` replaces a function, not a live Store object.
_STORE: Store | None = None
_EMBEDDER: Embedder | None = None


def _settings():
    return load_settings()


def _store() -> Store:
    global _STORE
    if _STORE is None:
        _STORE = Store(_settings().database_url)
    return _STORE


def _embedder() -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder()
    return _EMBEDDER


def _answer(question: str, hits, model: str):
    return answer_fn(question, hits, model=model)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    k: int | None = Field(default=None, ge=1, le=10)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "chunks": _store().count()}


@app.post("/api/chat")
def chat(body: ChatRequest, request: Request) -> dict:
    started = time.perf_counter()
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question is empty")

    settings = _settings()
    if not settings.anthropic_api_key:
        raise HTTPException(
            status_code=503,
            detail="No API key configured. Set ANTHROPIC_API_KEY on this server.",
        )

    # Prefer the header the platform sets itself; a client can send its own
    # x-forwarded-for, so treat the first value only as a fallback.
    ip = request.headers.get("x-real-ip") or request.headers.get("x-forwarded-for", "")
    ip = (ip or (request.client.host if request.client else "") or "unknown").split(",")[0].strip()
    allowed, remaining = check_and_consume(ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="This demo is rate limited. Clone the repo and run it with your own key.",
        )

    k = min(body.k or settings.top_k, 10)
    vector = _embedder().embed([question])[0]
    hits = _store().search(vector, k=k)
    result = _answer(question, hits, settings.model)
    return {
        "answer": result.text,
        "citations": [
            {"file": c.file, "page": c.page, "snippet": c.snippet, "marker": c.marker}
            for c in result.citations
        ],
        "model": settings.model,
        "remaining": remaining,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
