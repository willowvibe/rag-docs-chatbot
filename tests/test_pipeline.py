from pathlib import Path

from ragchat.ingest.pipeline import discover_documents, doc_id_for, ingest_path


def test_doc_id_is_the_stem():
    assert doc_id_for(Path("corpus/NIST.SP.800-63b.pdf")) == "nist-sp-800-63b"


def test_discovery_is_sorted_and_filters_by_extension(tmp_path):
    for name in ("b.md", "a.pdf", "c.txt", "d.html"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    found = discover_documents(tmp_path)
    assert [p.name for p in found] == ["a.pdf", "b.md", "d.html"]


def test_discovery_accepts_a_single_file(tmp_path):
    """`python -m ragchat.ingest.pipeline corpus/a.pdf` must ingest that file, not nothing."""
    doc = tmp_path / "a.md"
    doc.write_text("x", encoding="utf-8")
    assert discover_documents(doc) == [doc]


class _FakeEmbedder:
    dim = 384

    def embed(self, texts):
        return [[0.0] * self.dim for _ in texts]


class _FakeStore:
    def __init__(self):
        self.docs: dict[str, int] = {}

    def replace_doc(self, doc_id, chunks, vectors):
        assert len(chunks) == len(vectors)
        self.docs[doc_id] = len(chunks)
        return len(chunks)

    def count(self, doc_id=None):
        if doc_id is None:
            return sum(self.docs.values())
        return self.docs.get(doc_id, 0)


def test_ingest_path_stores_every_document(tmp_path):
    (tmp_path / "a.md").write_text("# Auth\n\n" + "Alpha text about authentication. " * 40, encoding="utf-8")
    (tmp_path / "b.md").write_text("# Limits\n\n" + "Beta text about rate limiting. " * 40, encoding="utf-8")
    store = _FakeStore()

    results = ingest_path(tmp_path, store, _FakeEmbedder())

    assert set(results) == {"a", "b"}
    assert all(n > 0 for n in results.values())
    assert store.count() == sum(results.values())


def test_reingesting_the_same_path_is_stable(tmp_path):
    (tmp_path / "a.md").write_text("# Auth\n\n" + "Alpha text about authentication. " * 40, encoding="utf-8")
    store = _FakeStore()

    first = ingest_path(tmp_path, store, _FakeEmbedder())
    second = ingest_path(tmp_path, store, _FakeEmbedder())

    assert first == second
    assert store.count() == first["a"]


def test_document_without_text_is_skipped(tmp_path, capsys):
    (tmp_path / "blank.md").write_text("   \n\n", encoding="utf-8")

    results = ingest_path(tmp_path, _FakeStore(), _FakeEmbedder())

    assert results == {}
    assert "skipped" in capsys.readouterr().err
