"""Turn a file into page-tagged text. Page numbers are 1-based and kept verbatim."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader


@dataclass(frozen=True)
class Page:
    doc_id: str
    file: str
    number: int
    text: str


def _clean(text: str) -> str:
    text = text.replace("­", "").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_pdf(path: Path, doc_id: str) -> list[Page]:
    reader = PdfReader(str(path))
    pages = []
    for i, raw in enumerate(reader.pages, start=1):
        text = _clean(raw.extract_text() or "")
        if text:
            pages.append(Page(doc_id=doc_id, file=str(path), number=i, text=text))
    return pages


def _parse_markdown(path: Path, doc_id: str) -> list[Page]:
    body = path.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^#\s+", body)
    pages: list[Page] = []
    for number, section in enumerate([s for s in sections if s.strip()], start=1):
        text = _clean(section)
        if text:
            pages.append(Page(doc_id=doc_id, file=str(path), number=number, text=text))
    return pages


def _parse_html(path: Path, doc_id: str) -> list[Page]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    chunks = soup.find_all("h1")
    pages: list[Page] = []
    if not chunks:
        text = _clean(soup.get_text("\n"))
        return [Page(doc_id=doc_id, file=str(path), number=1, text=text)] if text else []
    for number, heading in enumerate(chunks, start=1):
        parts = [heading.get_text(" ")]
        for node in heading.next_siblings:
            if getattr(node, "name", None) == "h1":
                break
            parts.append(node.get_text(" ") if hasattr(node, "get_text") else str(node))
        text = _clean("\n".join(parts))
        if text:
            pages.append(Page(doc_id=doc_id, file=str(path), number=number, text=text))
    return pages


_PARSERS = {".pdf": _parse_pdf, ".md": _parse_markdown, ".html": _parse_html, ".htm": _parse_html}


def parse_document(path: Path, doc_id: str) -> list[Page]:
    suffix = path.suffix.lower()
    parser = _PARSERS.get(suffix)
    if parser is None:
        raise ValueError(f"unsupported file type: {suffix}")
    return parser(path, doc_id)
