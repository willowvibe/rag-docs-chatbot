"""Every golden answer must be verifiable in the corpus, or the eval means nothing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ragchat.ingest.parsers import parse_document


def load_golden(golden_path: Path) -> list[dict]:
    """Shared by the validator and the runner so both read the format identically."""
    return [
        json.loads(line)
        for line in golden_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_golden(golden_path: Path, corpus_dir: Path) -> list[str]:
    items = load_golden(golden_path)
    pages_by_file: dict[str, dict[int, str]] = {}
    problems: list[str] = []
    for item in items:
        name = Path(item["file"]).name
        if name not in pages_by_file:
            path = corpus_dir / name
            if not path.exists():
                problems.append(f"{item['id']}: file not found: {path}")
                continue
            doc_id = path.stem.lower().replace(".", "-")
            pages_by_file[name] = {p.number: p.text.lower() for p in parse_document(path, doc_id)}
        page_text = pages_by_file[name].get(item["page"])
        if page_text is None:
            problems.append(f"{item['id']}: {name} has no page {item['page']}")
        elif item["evidence_phrase"].lower() not in page_text:
            problems.append(
                f"{item['id']}: phrase {item['evidence_phrase']!r} not on {name} page {item['page']}"
            )
    return problems


def main(argv: list[str]) -> int:
    golden = Path(argv[1] if len(argv) > 1 else "eval/golden.jsonl")
    problems = validate_golden(golden, Path("corpus"))
    for problem in problems:
        print(problem)
    print(f"{'FAIL' if problems else 'OK'}: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
