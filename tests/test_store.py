import os

import pytest

from ragchat.embeddings import Embedder
from ragchat.ingest.chunker import Chunk
from ragchat.store import Store

requires_db = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"), reason="needs a Postgres with pgvector"
)


@pytest.fixture
def store():
    # The same database is used by `make ingest` and by /api/health, so a test that
    # leaves rows behind corrupts every later chunk count. Clean on both sides.
    s = Store(os.environ["DATABASE_URL"])
    s.ensure_schema()
    s.replace_doc("d", [], [])
    yield s
    s.replace_doc("d", [], [])


def _chunk(i, text):
    return Chunk(doc_id="d", file="f.pdf", page=1, chunk_index=i, text=text, tokens=3)


@requires_db
def test_replace_doc_is_idempotent(store):
    chunks = [_chunk(0, "alpha"), _chunk(1, "beta")]
    vectors = [[0.0] * 383 + [1.0], [0.0] * 382 + [1.0, 0.0]]
    assert store.replace_doc("d", chunks, vectors) == 2
    assert store.replace_doc("d", chunks, vectors) == 2
    assert store.count("d") == 2


@requires_db
def test_search_returns_nearest_first(store):
    chunks = [_chunk(0, "near"), _chunk(1, "far")]
    vectors = [[1.0] + [0.0] * 383, [0.0] * 383 + [1.0]]
    store.replace_doc("d", chunks, vectors)
    hits = store.search([1.0] + [0.0] * 383, k=2)
    assert hits[0].text == "near"
    assert 0.99 <= hits[0].score <= 1.01


@requires_db
def test_hybrid_search_adds_keyword_matches(store):
    keyword_text = "zzhybridtoken keyword only"
    chunks = [_chunk(0, keyword_text), _chunk(1, "vector nearest")]
    vector = [1.0] + [0.0] * 383
    vectors = [[0.0] * 383 + [1.0], vector]
    store.replace_doc("d", chunks, vectors)

    hits = store.search(vector, k=2, query="zzhybridtoken", hybrid=True)

    assert keyword_text in [hit.text for hit in hits]
    assert all(hit.score > 0 for hit in hits)


@requires_db
def test_hybrid_search_with_only_stopwords_falls_back_to_vector(store):
    chunks = [_chunk(0, "zzhybridnear"), _chunk(1, "zzhybridfar")]
    vector = [1.0] + [0.0] * 383
    vectors = [vector, [0.0] * 383 + [1.0]]
    store.replace_doc("d", chunks, vectors)

    vector_only = store.search(vector, k=2)
    hybrid = store.search(vector, k=2, query="the and of", hybrid=True)

    assert [hit.text for hit in hybrid] == [hit.text for hit in vector_only]


@requires_db
def test_global_count(store):
    before = store.count()
    chunks = [_chunk(0, "alpha"), _chunk(1, "beta")]
    vectors = [[0.0] * 383 + [1.0], [0.0] * 382 + [1.0, 0.0]]
    store.replace_doc("d", chunks, vectors)
    assert store.count() == before + 2


def test_embedder_is_normalised_and_deterministic():
    emb = Embedder()
    a = emb.embed(["passwords must be salted"])
    emb.clear_cache()
    b = emb.embed(["passwords must be salted"])
    assert len(a[0]) == 384
    assert a == b
    norm = sum(x * x for x in a[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-3


def test_embedder_cache_returns_copies():
    emb = Embedder()
    first = emb.embed(["passwords must be salted"])
    expected = first[0][0]
    first[0][0] = expected + 123.0
    again = emb.embed(["passwords must be salted"])
    assert again[0][0] == expected
