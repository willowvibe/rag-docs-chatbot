from pathlib import Path

from ragchat.ingest.parsers import Page, parse_document

FIXTURES = Path(__file__).parent / "fixtures"


def test_pdf_pages_are_numbered_from_one():
    pages = parse_document(FIXTURES / "sample.pdf", "sample")
    assert [p.number for p in pages] == list(range(1, len(pages) + 1))
    assert all(p.file.endswith("sample.pdf") for p in pages)
    assert any("authentication" in p.text.lower() for p in pages)


def test_markdown_splits_on_top_level_headings(tmp_path):
    doc = tmp_path / "guide.md"
    doc.write_text("# One\n\nalpha text\n\n# Two\n\nbeta text\n", encoding="utf-8")
    pages = parse_document(doc, "guide")
    assert [p.number for p in pages] == [1, 2]
    assert "alpha" in pages[0].text
    assert "beta" in pages[1].text


def test_html_splits_on_h1(tmp_path):
    doc = tmp_path / "guide.html"
    doc.write_text("<html><body><h1>A</h1><p>first</p><h1>B</h1><p>second</p></body></html>",
                   encoding="utf-8")
    pages = parse_document(doc, "guide")
    assert len(pages) == 2
    assert "first" in pages[0].text and "second" in pages[1].text


def test_unknown_extension_raises(tmp_path):
    bad = tmp_path / "x.txt"
    bad.write_text("hello", encoding="utf-8")
    try:
        parse_document(bad, "x")
    except ValueError as exc:
        assert "unsupported" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")


def test_blank_pdf_pages_are_dropped():
    pages = parse_document(FIXTURES / "sample.pdf", "sample")
    assert 3 not in [p.number for p in pages]
    assert all(p.text.strip() for p in pages)
