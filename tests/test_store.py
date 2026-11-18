import os

import psycopg
import pytest

import ragchat.store as store_module
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


@requires_db
def test_list_documents_reports_pages_and_chunks_per_basename(store):
    chunks = [
        Chunk(
            doc_id="d",
            file="listdocs-alpha.pdf",
            page=1,
            chunk_index=0,
            text="a",
            tokens=1,
        ),
        Chunk(
            doc_id="d",
            file="listdocs-alpha.pdf",
            page=4,
            chunk_index=1,
            text="b",
            tokens=1,
        ),
        Chunk(
            doc_id="d",
            file="nested/listdocs-beta.pdf",
            page=2,
            chunk_index=0,
            text="c",
            tokens=1,
        ),
    ]
    vectors = [
        [0.0] * 383 + [1.0],
        [0.0] * 382 + [1.0, 0.0],
        [1.0] + [0.0] * 383,
    ]
    store.replace_doc("d", chunks, vectors)

    documents = {doc["file"]: doc for doc in store.list_documents()}

    assert documents["listdocs-alpha.pdf"] == {
        "file": "listdocs-alpha.pdf",
        "pages": 4,
        "chunks": 2,
    }
    assert documents["listdocs-beta.pdf"] == {
        "file": "listdocs-beta.pdf",
        "pages": 2,
        "chunks": 1,
    }


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0]


class _FakeConnection:
    def __init__(self, rows, fail_first=False):
        self.closed = False
        self.rows = rows
        self.fail_first = fail_first
        self.executed = 0

    def execute(self, sql, params=None):
        self.executed += 1
        if self.fail_first:
            self.fail_first = False
            raise psycopg.OperationalError("connection dropped")
        return _FakeCursor(self.rows)

    def close(self):
        self.closed = True


class _FakeConnect:
    def __init__(self, rows, fail_first=False):
        self.rows = rows
        self.fail_first = fail_first
        self.connections = []

    def __call__(self, *args, **kwargs):
        conn = _FakeConnection(self.rows, fail_first=self.fail_first and not self.connections)
        self.connections.append(conn)
        return conn


def _use_fake_connect(monkeypatch, fake):
    monkeypatch.setattr(store_module.psycopg, "connect", fake)
    monkeypatch.setattr(store_module, "register_vector", lambda conn: None)


def test_search_reuses_one_connection(monkeypatch):
    fake = _FakeConnect([("f.pdf", 1, 0, "text", 0.9)])
    _use_fake_connect(monkeypatch, fake)
    store = Store("postgresql://example")

    vector = [0.0] * 384
    first = store.search(vector)
    second = store.search(vector)

    assert first == second
    assert len(fake.connections) == 1
    assert fake.connections[0].executed == 2


def test_search_reconnects_after_dropped_connection(monkeypatch):
    fake = _FakeConnect([("f.pdf", 1, 0, "text", 0.9)], fail_first=True)
    _use_fake_connect(monkeypatch, fake)
    store = Store("postgresql://example")

    hits = store.search([0.0] * 384)

    assert hits[0].text == "text"
    assert len(fake.connections) == 2
    assert fake.connections[0].closed
    assert not fake.connections[1].closed


def test_count_reuses_the_search_connection(monkeypatch):
    fake = _FakeConnect([(2,)])
    _use_fake_connect(monkeypatch, fake)
    store = Store("postgresql://example")

    assert store.count() == 2
    assert store.count() == 2

    assert len(fake.connections) == 1


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
