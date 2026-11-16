"""parse -> chunk -> embed -> store, one document at a time, idempotently."""

from __future__ import annotations

import sys
from pathlib import Path

from ragchat.embeddings import Embedder
from ragchat.ingest.chunker import chunk_pages
from ragchat.ingest.parsers import parse_document
from ragchat.store import Store

SUPPORTED = {".pdf", ".md", ".html", ".htm"}


def doc_id_for(path: Path) -> str:
    return path.stem.lower().replace(".", "-")


def discover_documents(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED else []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED)


def ingest_path(path: Path, store: Store, embedder: Embedder) -> dict[str, int]:
    results: dict[str, int] = {}
    for doc in discover_documents(path):
        doc_id = doc_id_for(doc)
        pages = parse_document(doc, doc_id)
        chunks = chunk_pages(pages)
        if not chunks:
            print(f"{doc.name}: no extractable text, skipped", file=sys.stderr)
            continue
        vectors = embedder.embed([c.text for c in chunks])
        stored = store.replace_doc(doc_id, chunks, vectors)
        results[doc_id] = stored
        print(f"{doc.name}: {len(pages)} pages -> {stored} chunks")
    return results


def main(argv: list[str]) -> int:
    from ragchat.config import load_settings

    target = Path(argv[1]) if len(argv) > 1 else Path("corpus")
    if not target.exists():
        sys.exit(f"no such path: {target}")
    settings = load_settings()
    store = Store(settings.database_url)
    store.ensure_schema()
    results = ingest_path(target, store, Embedder())
    print(f"total={sum(results.values())} chunks across {len(results)} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
