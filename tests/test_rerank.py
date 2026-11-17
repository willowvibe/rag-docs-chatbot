from ragchat import rerank as rerank_module
from ragchat.rerank import Reranker
from ragchat.store import Hit


def _hit(text, page=1, score=0.5):
    return Hit(file="a.pdf", page=page, chunk_index=0, text=text, score=score)


class _FakeEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def rerank(self, query, documents):
        self.calls.append((query, list(documents)))
        return iter(self.scores)


def test_rerank_orders_by_score_and_replaces_the_hit_score():
    encoder = _FakeEncoder([0.1, 0.9, 0.5])
    reranker = Reranker(encoder=encoder)
    hits = [_hit("first"), _hit("second"), _hit("third")]

    reranked = reranker.rerank("query", hits, top_k=2)

    assert [hit.text for hit in reranked] == ["second", "third"]
    assert [hit.score for hit in reranked] == [0.9, 0.5]
    assert encoder.calls == [("query", ["first", "second", "third"])]


def test_rerank_is_stable_for_tied_scores():
    reranker = Reranker(encoder=_FakeEncoder([0.7, 0.7, 0.7]))
    hits = [_hit("a"), _hit("b"), _hit("c")]

    reranked = reranker.rerank("query", hits, top_k=3)

    assert [hit.text for hit in reranked] == ["a", "b", "c"]


def test_rerank_returns_at_most_top_k():
    reranker = Reranker(encoder=_FakeEncoder([0.4, 0.3, 0.2]))

    reranked = reranker.rerank("query", [_hit("a"), _hit("b"), _hit("c")], top_k=1)

    assert [hit.text for hit in reranked] == ["a"]


def test_rerank_empty_hits_returns_empty_without_loading_a_model():
    encoder = _FakeEncoder([])
    reranker = Reranker(encoder=encoder)

    assert reranker.rerank("query", [], top_k=5) == []
    assert encoder.calls == []


def test_rerank_model_is_loaded_lazily(monkeypatch):
    calls = []

    class _FakeTextCrossEncoder:
        def __init__(self, model_name):
            calls.append(model_name)

        def rerank(self, query, documents):
            return [0.5 for _ in documents]

    monkeypatch.setattr(rerank_module, "TextCrossEncoder", _FakeTextCrossEncoder)
    reranker = Reranker("some/model")

    assert calls == []
    reranker.rerank("query", [_hit("a")], top_k=1)
    assert calls == ["some/model"]
    reranker.rerank("query", [_hit("b")], top_k=1)
    assert calls == ["some/model"]
