"""hit@k, MRR, citation grounding, and an LLM-judged faithfulness score."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from ragchat.answer import answer as answer_fn
from ragchat.embeddings import Embedder
from ragchat.eval.validate_golden import EXPECTED_ITEMS, load_golden
from ragchat.store import Store

JUDGE_SYSTEM = """You grade whether an answer is supported by its sources.
Reply with one line: SCORE <0-3> then a short reason.
3 = every claim is supported by a source; 2 = mostly supported, one claim unsupported;
1 = partially supported; 0 = contradicted by the sources or refuses when the answer was present."""

# Anchor on the label, not on the first bare digit: a reason like "3 sources ..." must
# not be read as a score. Markdown emphasis is stripped before matching.
_JUDGE_SCORE_RE = re.compile(r"SCORE\s*[:\-]?\s*([0-3])\b", re.IGNORECASE)


def _judge(client, model: str, question: str, answer_text: str,
           sources: str) -> float | None:
    response = client.messages.create(
        model=model,
        max_tokens=200,
        temperature=0,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content":
                   f"QUESTION: {question}\n\nSOURCES:\n{sources}\n\nANSWER:\n{answer_text}"}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    cleaned = text.replace("*", "").replace("_", "").replace("`", "")
    match = _JUDGE_SCORE_RE.search(cleaned)
    if match is None:
        return None
    return int(match.group(1)) / 3.0


def run_eval(golden_path: Path, store: Store, embedder: Embedder, client,
             model: str, k: int = 5) -> dict:
    items = load_golden(golden_path)
    hits_ok = 0
    rr_total = 0.0
    grounded = 0
    faithfulness_total = 0.0
    judged = 0
    judge_unparsed = 0
    per_question = []

    for item in items:
        gold_name = Path(item["file"]).name
        vector = embedder.embed([item["question"]])[0]
        hits = store.search(vector, k=k)
        rank = next((i for i, h in enumerate(hits, start=1)
                     if Path(h.file).name == gold_name and h.page == item["page"]), None)
        hit = rank is not None
        hits_ok += int(hit)
        rr_total += 0.0 if rank is None else 1.0 / rank

        result = answer_fn(item["question"], hits, client=client, model=model)
        page_ok = any(c.page == item["page"] and Path(c.file).name == gold_name
                      for c in result.citations)
        grounded += int(page_ok)
        sources = "\n\n".join(f"[{i}] {h.file} page {h.page}\n{h.text}"
                              for i, h in enumerate(hits, start=1))
        score = _judge(client, model, item["question"], result.text, sources)
        if score is None:
            judge_unparsed += 1
        else:
            faithfulness_total += score
            judged += 1
        per_question.append({
            "id": item["id"], "hit": hit, "rank": rank, "cited_gold_page": page_ok,
            "faithfulness": None if score is None else round(score, 3),
            "answer": result.text,
        })

    n = len(items)
    return {
        "n": n,
        "k": k,
        "hit_at_k": round(hits_ok / n, 3) if n else 0.0,
        "mrr": round(rr_total / n, 3) if n else 0.0,
        "grounding_rate": round(grounded / n, 3) if n else 0.0,
        "faithfulness": round(faithfulness_total / judged, 3) if judged else None,
        "judge_unparsed": judge_unparsed,
        "per_question": per_question,
    }


def render_markdown(results: dict, model: str, file_counts: dict[str, int]) -> str:
    row_lines = []
    for q in results["per_question"]:
        faith = "-" if q["faithfulness"] is None else f"{q['faithfulness']:.2f}"
        row_lines.append(
            f"| {q['id']} | {'yes' if q['hit'] else 'no'} | {q['rank'] or '-'} | "
            f"{'yes' if q['cited_gold_page'] else 'no'} | {faith} |"
        )
    rows = "\n".join(row_lines)
    total_chunks = sum(file_counts.values())
    per_file = "; ".join(f"{name}: {count}" for name, count in sorted(file_counts.items()))
    faithfulness = results["faithfulness"]
    faithfulness_text = "n/a" if faithfulness is None else f"{faithfulness:.2f}"
    unparsed = results.get("judge_unparsed", 0)
    note = ""
    if unparsed:
        scored = results["n"] - unparsed
        note = (
            f"\n> {unparsed} judge reply(ies) could not be parsed and were excluded from the "
            f"faithfulness mean (scored questions: {scored}).\n"
        )
    return f"""# Retrieval and faithfulness eval

Corpus: {total_chunks} chunks across the golden files ({per_file}) · k = {results['k']} · answer model: `{model}` · questions: {results['n']}

| Metric | Value |
|---|---|
| Retrieval hit@{results['k']} | **{results['hit_at_k']:.1%}** |
| MRR | **{results['mrr']:.3f}** |
| Citation points at the gold page | **{results['grounding_rate']:.1%}** |
| Faithfulness (LLM judge, 0–1) | **{faithfulness_text}** |
| Judge replies unparsed | {unparsed} |
{note}
## Per question

| id | in top-k | rank | gold page cited | faithfulness |
|---|---|---|---|---|
{rows}

Reproduce with `make eval` (needs `ANTHROPIC_API_KEY` and an ingested corpus).
"""


def preflight(golden_path: Path, store: Store) -> list[str]:
    """Refuse to produce an all-zero report from an empty or partial setup."""
    try:
        items = load_golden(golden_path)
    except json.JSONDecodeError as exc:
        return [f"golden set is not valid JSON: {exc}"]
    except OSError as exc:
        return [f"could not read golden set {golden_path}: {exc}"]

    problems: list[str] = []
    if len(items) != EXPECTED_ITEMS:
        problems.append(
            f"golden set has {len(items)} item(s), expected {EXPECTED_ITEMS}"
        )
    if store.count() == 0:
        problems.append("store is empty; ingest the corpus before running the eval")
        return problems

    gold_files = sorted({
        Path(item["file"]).name
        for item in items
        if isinstance(item, dict) and isinstance(item.get("file"), str)
    })
    for name in gold_files:
        if store.count_for_file(name) == 0:
            problems.append(f"no chunks in the store for {name}")
    return problems


def main(argv: list[str]) -> int:
    from anthropic import Anthropic

    from ragchat.config import load_settings

    settings = load_settings()
    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY is not set; set ANTHROPIC_API_KEY to run the eval.",
              file=sys.stderr)
        return 1

    golden_path = Path(argv[1]) if len(argv) > 1 else Path("eval/golden.jsonl")
    store = Store(settings.database_url)
    problems = preflight(golden_path, store)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print("Preflight failed; no report written.", file=sys.stderr)
        return 1

    items = load_golden(golden_path)
    file_counts = {
        Path(item["file"]).name: store.count_for_file(Path(item["file"]).name)
        for item in items
    }
    results = run_eval(golden_path, store, Embedder(),
                       Anthropic(api_key=settings.anthropic_api_key),
                       settings.model, k=settings.top_k)
    Path("eval/report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    Path("eval/report.md").write_text(
        render_markdown(results, settings.model, file_counts), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "per_question"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
