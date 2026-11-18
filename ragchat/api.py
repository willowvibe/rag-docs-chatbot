"""One FastAPI app: JSON API plus the static chat page."""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ragchat.answer import answer as answer_fn
from ragchat.config import load_settings
from ragchat.embeddings import Embedder
from ragchat.llm import make_client
from ragchat.ratelimit import check_and_consume
from ragchat.rerank import Reranker
from ragchat.store import Store

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "corpus" / "manifest.json"

app = FastAPI(title="rag-docs-chatbot", docs_url="/api/docs", openapi_url="/api/openapi.json")

# The caches are module-private; the accessor functions are what the tests patch, so
# `monkeypatch.setattr(api, "_store", ...)` replaces a function, not a live Store object.
_STORE: Store | None = None
_EMBEDDER: Embedder | None = None
_RERANKER: Reranker | None = None
_DOCUMENTS_CACHE: tuple[Store, dict] | None = None


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


def _reranker() -> Reranker:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = Reranker(_settings().rerank_model)
    return _RERANKER


def _answer(question: str, hits, model: str, fallback_model: str | None = None):
    return answer_fn(
        question,
        hits,
        client=make_client(_settings()),
        model=model,
        fallback_model=fallback_model,
    )


def _load_manifest() -> dict:
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"documents": [], "examples": []}
    if not isinstance(data, dict):
        return {"documents": [], "examples": []}
    documents = data.get("documents")
    examples = data.get("examples")
    return {
        "documents": documents if isinstance(documents, list) else [],
        "examples": examples if isinstance(examples, list) else [],
    }


def _build_documents(store: Store) -> dict:
    manifest = _load_manifest()
    metadata = {
        item["file"]: item
        for item in manifest["documents"]
        if isinstance(item, dict) and isinstance(item.get("file"), str)
    }
    documents = []
    for row in store.list_documents():
        name = row["file"]
        meta = metadata.get(name)
        title = meta.get("title") if meta else None
        url = meta.get("url") if meta else None
        documents.append(
            {
                "file": name,
                "title": title if isinstance(title, str) else name,
                "url": url if isinstance(url, str) else None,
                "pages": row["pages"],
                "chunks": row["chunks"],
            }
        )
    return {"documents": documents, "examples": manifest["examples"]}


def _documents_payload() -> dict:
    global _DOCUMENTS_CACHE
    store = _store()
    if _DOCUMENTS_CACHE is None or _DOCUMENTS_CACHE[0] is not store:
        _DOCUMENTS_CACHE = (store, _build_documents(store))
    return _DOCUMENTS_CACHE[1]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int | None = Field(default=None, ge=1, le=10)


def _seconds_until_next_hour() -> int:
    return 3600 - (int(time.time()) % 3600)


def _seconds_until_utc_midnight() -> int:
    return 86400 - (int(time.time()) % 86400)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "chunks": _store().count()}


@app.get("/api/documents")
def documents() -> dict:
    return _documents_payload()


@app.post("/api/chat")
def chat(body: ChatRequest, request: Request) -> dict:
    started = time.perf_counter()
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question is empty")

    settings = _settings()
    if len(question) > settings.max_question_chars:
        raise HTTPException(
            status_code=422,
            detail=f"question is longer than {settings.max_question_chars} characters",
        )
    if not settings.llm_api_key:
        raise HTTPException(
            status_code=503,
            detail=f"No API key configured. Set {settings.llm_key_env} on this server.",
        )

    # Prefer the header the platform sets itself; a client can send its own
    # x-forwarded-for, so treat the first value only as a fallback.
    ip = request.headers.get("x-real-ip") or request.headers.get("x-forwarded-for", "")
    ip = (ip or (request.client.host if request.client else "") or "unknown").split(",")[0].strip()
    allowed, remaining, reason = check_and_consume(ip)
    if not allowed:
        if reason == "daily":
            resets_in_seconds = _seconds_until_utc_midnight()
        else:
            resets_in_seconds = _seconds_until_next_hour()
        raise HTTPException(
            status_code=429,
            detail={
                "message": (
                    "This demo is rate limited. Clone the repo and run it with your own key."
                ),
                "limit": settings.per_ip_per_hour,
                "resets_in_seconds": resets_in_seconds,
                "reason": reason,
            },
        )

    k = min(body.k or settings.top_k, 10)
    vector = _embedder().embed([question])[0]
    if settings.rerank_enabled:
        hits = _store().search(
            vector,
            k=settings.rerank_candidates,
            query=question,
            hybrid=settings.hybrid_enabled,
        )
        hits = _reranker().rerank(question, hits, k)
    else:
        hits = _store().search(
            vector, k=k, query=question, hybrid=settings.hybrid_enabled
        )
    result = _answer(question, hits, settings.model, settings.fallback_model)
    return {
        "answer": result.text,
        "citations": [
            {"file": c.file, "page": c.page, "snippet": c.snippet, "marker": c.marker}
            for c in result.citations
        ],
        "model": settings.model,
        "limit": settings.per_ip_per_hour if settings.rate_limit_enabled else None,
        "remaining": remaining,
        "resets_in_seconds": (
            _seconds_until_next_hour() if settings.rate_limit_enabled else None
        ),
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
