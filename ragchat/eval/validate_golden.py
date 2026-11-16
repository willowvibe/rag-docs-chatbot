"""Every golden answer must be verifiable in the corpus, or the eval means nothing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ragchat.ingest.parsers import parse_document

EXPECTED_ITEMS = 30

_REQUIRED_FIELDS = ("id", "question", "file", "page", "evidence_phrase")
_EXPECTED_TYPES: dict[str, type] = {
    "id": str,
    "question": str,
    "file": str,
    "page": int,
    "evidence_phrase": str,
}
_TYPE_NAMES = {"id": "a string", "question": "a string", "file": "a string",
               "page": "an integer", "evidence_phrase": "a string"}


def load_golden(golden_path: Path) -> list[dict]:
    """Strict loader for the runner; the validator reports malformed lines instead."""
    return [
        json.loads(line)
        for line in golden_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _type_ok(value, expected: type) -> bool:
    # bool is a subclass of int, but `true` is not a page number.
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected)


def _read_items(golden_path: Path) -> tuple[list[dict], list[str], int]:
    items: list[dict] = []
    problems: list[str] = []
    parsed_count = 0
    seen_ids: set[str] = set()
    for lineno, line in enumerate(
        golden_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"line {lineno}: malformed JSON: {exc.msg}")
            continue
        parsed_count += 1
        if not isinstance(item, dict):
            problems.append(f"line {lineno}: expected a JSON object")
            continue
        item_id = item.get("id")
        label = item_id if isinstance(item_id, str) and item_id else f"line {lineno}"
        missing = [field for field in _REQUIRED_FIELDS if field not in item]
        if missing:
            problems.append(f"{label}: missing required field(s): {', '.join(missing)}")
            continue
        bad = [field for field, expected in _EXPECTED_TYPES.items()
               if not _type_ok(item[field], expected)]
        if bad:
            details = ", ".join(f"{field} must be {_TYPE_NAMES[field]}" for field in bad)
            problems.append(f"{label}: wrong type: {details}")
            continue
        if item["id"] in seen_ids:
            problems.append(f"{label}: duplicate id")
        else:
            seen_ids.add(item["id"])
        items.append(item)
    return items, problems, parsed_count


def validate_golden(golden_path: Path, corpus_dir: Path,
                    expected_items: int = EXPECTED_ITEMS) -> list[str]:
    items, problems, parsed_count = _read_items(golden_path)
    if parsed_count != expected_items:
        problems.append(f"expected {expected_items} items, found {parsed_count}")
    if parsed_count == 0:
        # An empty set already fails the count check; listing every corpus file as
        # unrepresented would only add noise.
        return problems

    required_files = {path.name for path in corpus_dir.glob("*.pdf")}
    represented = {Path(item["file"]).name for item in items}
    for name in sorted(required_files - represented):
        problems.append(f"corpus file not represented in golden set: {name}")

    pages_by_file: dict[str, dict[int, str]] = {}
    missing_files: set[str] = set()
    for item in items:
        name = Path(item["file"]).name
        if name in missing_files:
            continue
        if name not in pages_by_file:
            path = corpus_dir / name
            if not path.exists():
                problems.append(f"{item['id']}: file not found: {path}")
                missing_files.add(name)
                continue
            doc_id = path.stem.lower().replace(".", "-")
            pages_by_file[name] = {
                p.number: p.text.lower() for p in parse_document(path, doc_id)
            }
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
