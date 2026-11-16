"""hit@k, MRR, citation grounding, and an LLM-judged faithfulness score."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ragchat.answer import answer as answer_fn
from ragchat.embeddings import Embedder
from ragchat.eval.validate_golden import load_golden
from ragchat.store import Store

JUDGE_SYSTEM = """You grade whether an answer is supported by its sources.
Reply with one line: SCORE <0-3> then a short reason.
3 = every claim is supported by a source; 2 = mostly supported, one claim unsupported;
1 = partially supported; 0 = contradicted by the sources or refuses when the answer was present."""


def _judge(client, model: str, question: str, answer_text: str, sources: str) -> float:
    response = client.messages.create(
        model=model,
        max_tokens=200,
        temperature=0,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content":
                   f"QUESTION: {question}\n\nSOURCES:\n{sources}\n\nANSWER:\n{answer_text}"}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    for token in text.split():
        if token in {"0", "1", "2", "3"}:
            return int(token) / 3.0
    return 0.0


def run_eval(golden_path: Path, store: Store, embedder: Embedder, client,
             model: str, k: int = 5) -> dict:
    items = load_golden(golden_path)
    hits_ok = 0
    rr_total = 0.0
    grounded = 0
    faithfulness_total = 0.0
    per_question = []

    for item in items:
        vector = embedder.embed([item["question"]])[0]
        hits = store.search(vector, k=k)
        rank = next((i for i, h in enumerate(hits, start=1)
                     if h.file.endswith(Path(item["file"]).name) and h.page == item["page"]), None)
        hit = rank is not None
        hits_ok += int(hit)
        rr_total += 0.0 if rank is None else 1.0 / rank

        result = answer_fn(item["question"], hits, client=client, model=model)
        page_ok = any(c.page == item["page"] and Path(c.file).name == Path(item["file"]).name
                      for c in result.citations)
        grounded += int(page_ok)
        sources = "\n\n".join(f"[{i}] {h.file} page {h.page}\n{h.text}"
                              for i, h in enumerate(hits, start=1))
        score = _judge(client, model, item["question"], result.text, sources)
        faithfulness_total += score
        per_question.append({
            "id": item["id"], "hit": hit, "rank": rank, "cited_gold_page": page_ok,
            "faithfulness": round(score, 3), "answer": result.text,
        })

    n = len(items)
    return {
        "n": n,
        "k": k,
        "hit_at_k": round(hits_ok / n, 3) if n else 0.0,
        "mrr": round(rr_total / n, 3) if n else 0.0,
        "grounding_rate": round(grounded / n, 3) if n else 0.0,
        "faithfulness": round(faithfulness_total / n, 3) if n else 0.0,
        "per_question": per_question,
    }


def render_markdown(results: dict, model: str, chunks: int) -> str:
    rows = "\n".join(
        f"| {q['id']} | {'yes' if q['hit'] else 'no'} | {q['rank'] or '-'} | "
        f"{'yes' if q['cited_gold_page'] else 'no'} | {q['faithfulness']:.2f} |"
        for q in results["per_question"]
    )
    return f"""# Retrieval and faithfulness eval

Corpus: {chunks} chunks · k = {results['k']} · answer model: `{model}` · questions: {results['n']}

| Metric | Value |
|---|---|
| Retrieval hit@{results['k']} | **{results['hit_at_k']:.1%}** |
| MRR | **{results['mrr']:.3f}** |
| Citation points at the gold page | **{results['grounding_rate']:.1%}** |
| Faithfulness (LLM judge, 0–1) | **{results['faithfulness']:.2f}** |

## Per question

| id | in top-k | rank | gold page cited | faithfulness |
|---|---|---|---|---|
{rows}

Reproduce with `make eval` (needs `ANTHROPIC_API_KEY` and an ingested corpus).
"""


def main(argv: list[str]) -> int:
    from anthropic import Anthropic

    from ragchat.config import load_settings

    settings = load_settings()
    store = Store(settings.database_url)
    results = run_eval(Path("eval/golden.jsonl"), store, Embedder(), Anthropic(),
                       settings.model, k=settings.top_k)
    Path("eval/report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    Path("eval/report.md").write_text(
        render_markdown(results, settings.model, store.count()), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "per_question"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
