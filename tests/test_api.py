from dataclasses import replace

from fastapi.testclient import TestClient

from ragchat import api
from ragchat.answer import Answer, Citation
from ragchat.config import load_settings
from ragchat.store import Hit


class FakeStore:
    def count(self):
        return 42

    def search(self, vector, k=5):
        return [Hit(file="a.pdf", page=3, chunk_index=0, text="eight characters", score=0.9)]


def _client(monkeypatch, reply="At least 8 characters [1]."):
    # Only the tests that reach the chat route need a key; patch the accessor so the
    # value is deterministic even when a developer `.env` has a real key.
    monkeypatch.setattr(
        api, "_settings", lambda: replace(load_settings(), anthropic_api_key="sk-test")
    )
    monkeypatch.setattr(api, "_store", lambda: FakeStore())
    monkeypatch.setattr(api, "_embedder", lambda: type("E", (), {"embed": lambda self, t: [[0.0] * 384]})())
    monkeypatch.setattr(
        api,
        "_answer",
        lambda q, hits, model: Answer(
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
            load_settings(), anthropic_api_key="sk-test", max_question_chars=10
        ),
    )
    response = client.post("/api/chat", json={"question": "x" * 11})
    assert response.status_code == 422


def test_chat_returns_503_without_a_key(monkeypatch):
    """Patch the settings accessor, not the environment: `load_settings()` also reads
    `.env`, so deleting the env var alone would not guarantee an empty key."""
    client = _client(monkeypatch)
    monkeypatch.setattr(
        api, "_settings", lambda: replace(load_settings(), anthropic_api_key="")
    )
    assert client.post("/api/chat", json={"question": "q"}).status_code == 503


def test_index_page_is_served(monkeypatch):
    assert _client(monkeypatch).get("/").status_code == 200
