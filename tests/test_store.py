import os

import pytest

from ragchat.embeddings import Embedder
from ragchat.ingest.chunker import Chunk
from ragchat.store import Store

pytestmark = pytest.mark.skipif(
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


def test_replace_doc_is_idempotent(store):
    chunks = [_chunk(0, "alpha"), _chunk(1, "beta")]
    vectors = [[0.0] * 383 + [1.0], [0.0] * 382 + [1.0, 0.0]]
    assert store.replace_doc("d", chunks, vectors) == 2
    assert store.replace_doc("d", chunks, vectors) == 2
    assert store.count("d") == 2


def test_search_returns_nearest_first(store):
    chunks = [_chunk(0, "near"), _chunk(1, "far")]
    vectors = [[1.0] + [0.0] * 383, [0.0] * 383 + [1.0]]
    store.replace_doc("d", chunks, vectors)
    hits = store.search([1.0] + [0.0] * 383, k=2)
    assert hits[0].text == "near"
    assert 0.99 <= hits[0].score <= 1.01


def test_embedder_is_normalised_and_deterministic():
    emb = Embedder()
    a = emb.embed(["passwords must be salted"])
    b = emb.embed(["passwords must be salted"])
    assert len(a[0]) == 384
    assert a == b
    norm = sum(x * x for x in a[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-3
