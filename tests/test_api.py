from dataclasses import replace

from fastapi.testclient import TestClient

from ragchat import api
from ragchat.answer import Answer, Citation
from ragchat.config import load_settings
from ragchat.store import Hit


class FakeStore:
    def count(self):
        return 42

    def search(self, vector, k=5, query=None, hybrid=False):
        return [Hit(file="a.pdf", page=3, chunk_index=0, text="eight characters", score=0.9)]


def _client(monkeypatch, reply="At least 8 characters [1]."):
    # Only the tests that reach the chat route need a key; patch the accessor so the
    # value is deterministic even when a developer `.env` has a real key.
    monkeypatch.setattr(
        api,
        "_settings",
        lambda: replace(
            load_settings(), llm_provider="ollama", ollama_api_key="sk-test"
        ),
    )
    monkeypatch.setattr(api, "_store", lambda: FakeStore())
    monkeypatch.setattr(api, "_embedder", lambda: type("E", (), {"embed": lambda self, t: [[0.0] * 384]})())
    monkeypatch.setattr(
        api,
        "_answer",
        lambda q, hits, model, fallback_model=None: Answer(
            text=reply,
            citations=[
                Citation(file="a.pdf", page=3, snippet="eight characters", marker=1)
            ],
            used_hits=hits,
        ),
    )
    return TestClient(api.app)


def test_health_reports_chunk_count(monkeypatch):
    assert _client(monkeypatch).get("/api/health").json() == {"status": "ok", "chunks": 42}


def test_chat_returns_answer_and_citations(monkeypatch):
    body = _client(monkeypatch).post("/api/chat", json={"question": "How long?"}).json()
    assert body["answer"] == "At least 8 characters [1]."
    assert body["citations"][0]["page"] == 3
    assert body["citations"][0]["file"] == "a.pdf"
    assert body["citations"][0]["snippet"] == "eight characters"
    assert body["citations"][0]["marker"] == 1
    assert "elapsed_ms" in body


def test_chat_with_rerank_searches_candidates_and_passes_reranked_top_k_to_answer(
    monkeypatch,
):
    client = _client(monkeypatch)
    hits = [
        Hit(file=f"{name}.pdf", page=i, chunk_index=0, text=name, score=0.5)
        for i, name in enumerate(["a", "b", "c"], start=1)
    ]
    searched = {}
    reranked = {}
    answered = {}

    class RecordingStore:
        def count(self):
            return 42

        def search(self, vector, k=5, query=None, hybrid=False):
            searched["k"] = k
            searched["query"] = query
            searched["hybrid"] = hybrid
            return hits

    class FakeReranker:
        def rerank(self, query, candidates, top_k):
            reranked["query"] = query
            reranked["candidates"] = list(candidates)
            reranked["top_k"] = top_k
            return list(reversed(candidates))[:top_k]

    def fake_answer(question, selected, model, fallback_model=None):
        answered["hits"] = list(selected)
        return Answer(text="ok", citations=[], used_hits=selected)

    monkeypatch.setattr(
        api,
        "_settings",
        lambda: replace(
            load_settings(),
            llm_provider="ollama",
            ollama_api_key="sk-test",
            rerank_enabled=True,
            rerank_candidates=20,
            top_k=5,
            hybrid_enabled=True,
        ),
    )
    monkeypatch.setattr(api, "_store", lambda: RecordingStore())
    monkeypatch.setattr(api, "_reranker", lambda: FakeReranker())
    monkeypatch.setattr(api, "_answer", fake_answer)

    response = client.post("/api/chat", json={"question": "How long?", "k": 2})

    assert response.status_code == 200
    assert searched["k"] == 20
    assert searched["query"] == "How long?"
    assert searched["hybrid"] is True
    assert reranked["query"] == "How long?"
    assert reranked["candidates"] == hits
    assert reranked["top_k"] == 2
    assert answered["hits"] == list(reversed(hits))[:2]


def test_chat_passes_query_and_hybrid_flag_to_search(monkeypatch):
    client = _client(monkeypatch)
    searched = {}

    class RecordingStore:
        def count(self):
            return 42

        def search(self, vector, k=5, query=None, hybrid=False):
            searched["k"] = k
            searched["query"] = query
            searched["hybrid"] = hybrid
            return [Hit(file="a.pdf", page=3, chunk_index=0, text="t", score=0.9)]

    monkeypatch.setattr(
        api,
        "_settings",
        lambda: replace(
            load_settings(),
            llm_provider="ollama",
            ollama_api_key="sk-test",
            hybrid_enabled=True,
            top_k=5,
        ),
    )
    monkeypatch.setattr(api, "_store", lambda: RecordingStore())

    response = client.post("/api/chat", json={"question": "How long?"})

    assert response.status_code == 200
    assert searched == {"k": 5, "query": "How long?", "hybrid": True}


def test_chat_returns_the_rate_limit_detail_when_blocked(monkeypatch):
    client = _client(monkeypatch)
    monkeypatch.setattr(api, "check_and_consume", lambda ip: (False, 0))
    response = client.post("/api/chat", json={"question": "q"})
    assert response.status_code == 429
    assert (
        response.json()["detail"]
        == "This demo is rate limited. Clone the repo and run it with your own key."
    )


def test_chat_rejects_out_of_range_k(monkeypatch):
    client = _client(monkeypatch)
    assert client.post("/api/chat", json={"question": "q", "k": -1}).status_code == 422
    assert client.post("/api/chat", json={"question": "q", "k": 11}).status_code == 422


def test_chat_rejects_empty_and_overlong_questions(monkeypatch):
    client = _client(monkeypatch)
    assert client.post("/api/chat", json={"question": "   "}).status_code == 422
    assert client.post("/api/chat", json={"question": "x" * 501}).status_code == 422


def test_chat_rejects_question_longer_than_configured_limit(monkeypatch):
    client = _client(monkeypatch)
    monkeypatch.setattr(
        api,
        "_settings",
        lambda: replace(
            load_settings(),
            llm_provider="ollama",
            ollama_api_key="sk-test",
            max_question_chars=10,
        ),
    )
    response = client.post("/api/chat", json={"question": "x" * 11})
    assert response.status_code == 422


def test_chat_returns_503_without_a_key(monkeypatch):
    """Patch the settings accessor, not the environment: `load_settings()` also reads
    `.env`, so deleting the env var alone would not guarantee an empty key."""
    client = _client(monkeypatch)
    monkeypatch.setattr(
        api,
        "_settings",
        lambda: replace(
            load_settings(),
            llm_provider="ollama",
            ollama_api_key="",
            anthropic_api_key="",
        ),
    )
    response = client.post("/api/chat", json={"question": "q"})
    assert response.status_code == 503
    assert "OLLAMA_API_KEY" in response.json()["detail"]


def test_answer_seam_receives_the_made_client(monkeypatch):
    sentinel = object()
    captured = {}

    def fake_answer(question, hits, client=None, model=None, fallback_model=None):
        captured["client"] = client
        return Answer(text="ok", citations=[], used_hits=hits)

    monkeypatch.setattr(api, "make_client", lambda settings: sentinel)
    monkeypatch.setattr(api, "answer_fn", fake_answer)

    api._answer("q", [], "m")

    assert captured["client"] is sentinel


def test_index_page_is_served(monkeypatch):
    assert _client(monkeypatch).get("/").status_code == 200
