import tiktoken

from ragchat.ingest.chunker import chunk_pages
from ragchat.ingest.parsers import Page

ENC = tiktoken.get_encoding("cl100k_base")


def _pages(texts, number_start=1):
    return [Page(doc_id="d", file="f.pdf", number=number_start + i, text=t)
            for i, t in enumerate(texts)]


def test_short_page_is_one_chunk():
    chunks = chunk_pages(_pages(["a short page about passwords"]))
    assert len(chunks) == 1
    assert chunks[0].page == 1
    assert chunks[0].chunk_index == 0


def test_long_page_splits_with_overlap():
    words = " ".join(f"w{i}" for i in range(1200))
    chunks = chunk_pages(_pages([words]), max_tokens=100, overlap=20)
    assert len(chunks) > 1
    assert all(c.tokens <= 100 for c in chunks)
    first_tail = ENC.encode(chunks[0].text)[-20:]
    second_head = ENC.encode(chunks[1].text)[:20]
    assert first_tail == second_head


def test_chunk_index_is_document_wide_and_monotonic():
    chunks = chunk_pages(_pages(["one two three", "four five six"]))
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[-1].page == 2


def test_chunking_is_deterministic():
    pages = _pages([" ".join(f"t{i}" for i in range(900))])
    a = chunk_pages(pages, max_tokens=120, overlap=30)
    b = chunk_pages(pages, max_tokens=120, overlap=30)
    assert [(c.page, c.chunk_index, c.text) for c in a] == \
           [(c.page, c.chunk_index, c.text) for c in b]


def test_empty_pages_produce_nothing():
    assert chunk_pages(_pages(["", "   "])) == []


def test_overlap_must_be_smaller_than_window():
    try:
        chunk_pages(_pages(["x"]), max_tokens=10, overlap=10)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("expected ValueError")
