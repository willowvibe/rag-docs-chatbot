"""Fixed-size token windows with overlap, computed inside a single page."""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from ragchat.ingest.parsers import Page

_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    file: str
    page: int
    chunk_index: int
    text: str
    tokens: int


def chunk_pages(
    pages: list[Page], max_tokens: int = 400, overlap: int = 80
) -> list[Chunk]:
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")

    chunks: list[Chunk] = []
    index = 0
    step = max_tokens - overlap
    for page in pages:
        tokens = _ENC.encode(page.text)
        if not tokens:
            continue
        start = 0
        while start < len(tokens):
            window = tokens[start : start + max_tokens]
            # Keep the decoded text verbatim: stripping a mid-page window's
            # leading space changes its first token and breaks exact overlap.
            text = _ENC.decode(window)
            if text.strip():
                chunks.append(
                    Chunk(
                        doc_id=page.doc_id,
                        file=page.file,
                        page=page.number,
                        chunk_index=index,
                        text=text,
                        tokens=len(window),
                    )
                )
                index += 1
            if start + max_tokens >= len(tokens):
                break
            start += step
    return chunks
