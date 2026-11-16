from pathlib import Path

from ragchat.ingest.parsers import parse_document

FIXTURES = Path(__file__).parent / "fixtures"


def test_pdf_pages_are_numbered_from_one():
    pages = parse_document(FIXTURES / "sample.pdf", "sample")
    assert pages[0].number == 1
    assert [p.number for p in pages] == sorted(p.number for p in pages)
    assert all(p.file.endswith("sample.pdf") for p in pages)
    assert any("authentication" in p.text.lower() for p in pages)
    assert [p.number for p in pages if "authentication" in p.text.lower()] == [2]


def test_blank_page_mid_document_preserves_source_numbering(tmp_path):
    """A dropped blank page must leave a GAP in numbering, not renumber survivors."""
    from pypdf import PdfReader, PdfWriter

    source = PdfReader(str(FIXTURES / "sample.pdf"))
    writer = PdfWriter()
    writer.add_page(source.pages[0])
    writer.add_blank_page(
        width=float(source.pages[0].mediabox.width),
        height=float(source.pages[0].mediabox.height),
    )
    writer.add_page(source.pages[1])
    out = tmp_path / "gap.pdf"
    with open(out, "wb") as fh:
        writer.write(fh)

    pages = parse_document(out, "gap")
    assert [p.number for p in pages] == [1, 3]


def test_markdown_splits_on_top_level_headings(tmp_path):
    doc = tmp_path / "guide.md"
    doc.write_text("# One\n\nalpha text\n\n# Two\n\nbeta text\n", encoding="utf-8")
    pages = parse_document(doc, "guide")
    assert [p.number for p in pages] == [1, 2]
    assert "alpha" in pages[0].text
    assert "beta" in pages[1].text
    assert "beta" not in pages[0].text
    assert "alpha" not in pages[1].text


def test_markdown_preamble_becomes_the_first_section(tmp_path):
    doc = tmp_path / "g.md"
    doc.write_text("intro text\n\n# One\n\nalpha\n", encoding="utf-8")
    pages = parse_document(doc, "g")
    assert [p.number for p in pages] == [1, 2]
    assert "intro" in pages[0].text
    assert "alpha" in pages[1].text
    assert "intro" not in pages[1].text


def test_html_splits_on_h1(tmp_path):
    doc = tmp_path / "guide.html"
    doc.write_text("<html><body><h1>A</h1><p>first</p><h1>B</h1><p>second</p></body></html>",
                   encoding="utf-8")
    pages = parse_document(doc, "guide")
    assert len(pages) == 2
    assert "first" in pages[0].text and "second" in pages[1].text
    assert "second" not in pages[0].text
    assert "first" not in pages[1].text


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
