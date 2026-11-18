"""hit@k, MRR, citation grounding, and an LLM-judged faithfulness score."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import psycopg

from ragchat.answer import answer as answer_fn
from ragchat.embeddings import Embedder
from ragchat.eval.validate_golden import EXPECTED_ITEMS, load_golden
from ragchat.rerank import Reranker
from ragchat.store import Store

JUDGE_SYSTEM = """You grade whether an answer is supported by its sources.
Reply with one line: SCORE <0-3> then a short reason.
3 = every claim is supported by a source; 2 = mostly supported, one claim unsupported;
1 = partially supported; 0 = contradicted by the sources or refuses when the answer was present."""

# Anchor on the label, not on the first bare digit: a reason like "3 sources ..." must
# not be read as a score. Markdown emphasis is stripped before matching.
_JUDGE_SCORE_RE = re.compile(r"SCORE\s*[:\-]?\s*([0-3])\b", re.IGNORECASE)


def _normalize_for_match(text: str) -> str:
    """Join hyphenated line breaks, collapse whitespace, and case-fold."""
    joined = re.sub(r"-\s*\n\s*", "", text)
    return re.sub(r"\s+", " ", joined).strip().casefold()


def contains_evidence(text: str, phrase: str) -> bool:
    return _normalize_for_match(phrase) in _normalize_for_match(text)


def _citation_hit(citation, used_hits):
    """Map a Citation back to the hit that produced it."""
    if 1 <= citation.marker <= len(used_hits):
        hit = used_hits[citation.marker - 1]
        if hit.file == citation.file and hit.page == citation.page:
            return hit
    return next(
        (hit for hit in used_hits
         if hit.file == citation.file and hit.page == citation.page),
        None,
    )


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
             model: str, k: int = 5, reranker=None,
             candidates: int | None = None, hybrid: bool = False) -> dict:
    items = load_golden(golden_path)
    hits_ok = 0
    rr_total = 0.0
    grounded = 0
    evidence_hits_ok = 0
    evidence_rr_total = 0.0
    evidence_grounded = 0
    faithfulness_total = 0.0
    judged = 0
    judge_unparsed = 0
    per_question = []

    for item in items:
        gold_name = Path(item["file"]).name
        phrase = item["evidence_phrase"]
        vector = embedder.embed([item["question"]])[0]
        search_k = candidates if reranker is not None and candidates is not None else k
        hits = store.search(
            vector, k=search_k, query=item["question"], hybrid=hybrid
        )
        if reranker is not None:
            hits = reranker.rerank(item["question"], hits, k)
        rank = next((i for i, h in enumerate(hits, start=1)
                     if Path(h.file).name == gold_name and h.page == item["page"]), None)
        hit = rank is not None
        hits_ok += int(hit)
        rr_total += 0.0 if rank is None else 1.0 / rank

        evidence_rank = next((i for i, h in enumerate(hits, start=1)
                              if contains_evidence(h.text, phrase)), None)
        evidence_hits_ok += int(evidence_rank is not None)
        evidence_rr_total += (
            0.0 if evidence_rank is None else 1.0 / evidence_rank
        )

        result = answer_fn(
            item["question"], hits, client=client, model=model, fallbacks=[]
        )
        page_ok = any(c.page == item["page"] and Path(c.file).name == gold_name
                      for c in result.citations)
        grounded += int(page_ok)
        cited_evidence = any(
            hit is not None and contains_evidence(hit.text, phrase)
            for hit in (_citation_hit(c, result.used_hits) for c in result.citations)
        )
        evidence_grounded += int(cited_evidence)
        sources = "\n\n".join(f"[{i}] {h.file} page {h.page}\n{h.text}"
                              for i, h in enumerate(hits, start=1))
        score = _judge(client, model, item["question"], result.text, sources)
        if score is None:
            judge_unparsed += 1
        else:
            faithfulness_total += score
            judged += 1
        per_question.append({
            "id": item["id"], "hit": hit, "rank": rank,
            "evidence_rank": evidence_rank, "cited_gold_page": page_ok,
            "cited_evidence": cited_evidence,
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
        "evidence_hit_at_k": round(evidence_hits_ok / n, 3) if n else 0.0,
        "evidence_mrr": round(evidence_rr_total / n, 3) if n else 0.0,
        "evidence_grounding_rate": round(evidence_grounded / n, 3) if n else 0.0,
        "faithfulness": round(faithfulness_total / judged, 3) if judged else None,
        "judge_unparsed": judge_unparsed,
        "per_question": per_question,
    }


def render_markdown(results: dict, model: str, file_counts: dict[str, int],
                    reranker_model: str | None = None,
                    hybrid: bool = False) -> str:
    row_lines = []
    for q in results["per_question"]:
        faith = "-" if q["faithfulness"] is None else f"{q['faithfulness']:.2f}"
        row_lines.append(
            f"| {q['id']} | {'yes' if q['hit'] else 'no'} | {q['rank'] or '-'} | "
            f"{'yes' if q['cited_gold_page'] else 'no'} | {faith} | "
            f"{'yes' if q['evidence_rank'] else 'no'} | "
            f"{q['evidence_rank'] or '-'} | "
            f"{'yes' if q['cited_evidence'] else 'no'} |"
        )
    rows = "\n".join(row_lines)
    total_chunks = sum(file_counts.values())
    per_file = "; ".join(f"{name}: {count}" for name, count in sorted(file_counts.items()))
    faithfulness = results["faithfulness"]
    faithfulness_text = "n/a" if faithfulness is None else f"{faithfulness:.2f}"
    retrieval_mode = "hybrid" if hybrid else "vector"
    unparsed = results.get("judge_unparsed", 0)
    note = ""
    if unparsed:
        scored = results["n"] - unparsed
        note = (
            f"\n> {unparsed} judge reply(ies) could not be parsed and were excluded from the "
            f"faithfulness mean (scored questions: {scored}).\n"
        )
    difference = (
        f"Evidence metrics count any retrieved or cited passage that contains the question's "
        f"evidence phrase; the page-based metrics — Retrieval hit@{results['k']} "
        f"(strict: exact gold page), MRR (strict: exact gold page), and Citation points at "
        f"the gold page (strict: exact gold page) — only credit the exact gold page."
    )
    return f"""# Retrieval and faithfulness eval

Corpus: {total_chunks} chunks across the golden files ({per_file}) · k = {results['k']} · retrieval: {retrieval_mode} · reranker: {f'`{reranker_model}`' if reranker_model else 'none'} · answer model: `{model}` · questions: {results['n']}

| Metric | Value | Basis |
|---|---|---|
| Evidence hit@{results['k']} | **{results['evidence_hit_at_k']:.1%}** | any passage containing the evidence phrase |
| Evidence MRR | **{results['evidence_mrr']:.3f}** | any passage containing the evidence phrase |
| Citation contains evidence | **{results['evidence_grounding_rate']:.1%}** | any cited passage containing the evidence phrase |
| Retrieval hit@{results['k']} | **{results['hit_at_k']:.1%}** | (strict: exact gold page) |
| MRR | **{results['mrr']:.3f}** | (strict: exact gold page) |
| Citation points at the gold page | **{results['grounding_rate']:.1%}** | (strict: exact gold page) |
| Faithfulness (LLM judge, 0–1) | **{faithfulness_text}** | LLM judge |
| Judge replies unparsed | {unparsed} | judge parse status |
{note}
{difference}

## Per question

| id | in top-k | rank | gold page cited | faithfulness | evidence in top-k | evidence rank | evidence cited |
|---|---|---|---|---|---|---|---|
{rows}

Reproduce with `make eval` (needs `OLLAMA_API_KEY` (or `ANTHROPIC_API_KEY` with LLM_PROVIDER=anthropic) and an ingested corpus).
"""


INGEST_HINT = "run `make up && make ingest` to prepare the database, then retry"


def _db_problem(exc: psycopg.Error) -> str:
    detail = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    return f"cannot read the chunk store ({detail}); {INGEST_HINT}"


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
    try:
        total = store.count()
    except (psycopg.OperationalError, psycopg.errors.UndefinedTable) as exc:
        return [_db_problem(exc)]
    except psycopg.Error as exc:
        return [_db_problem(exc)]
    if total == 0:
        problems.append(f"store is empty; {INGEST_HINT}")
        return problems

    gold_files = sorted({
        Path(item["file"]).name
        for item in items
        if isinstance(item, dict) and isinstance(item.get("file"), str)
    })
    for name in gold_files:
        try:
            count = store.count_for_file(name)
        except (psycopg.OperationalError, psycopg.errors.UndefinedTable) as exc:
            return [_db_problem(exc)]
        except psycopg.Error as exc:
            return [_db_problem(exc)]
        if count == 0:
            problems.append(f"no chunks in the store for {name}")
    return problems


def main(argv: list[str]) -> int:
    from ragchat.config import DEFAULT_RERANK_MODEL, load_settings
    from ragchat.llm import make_client

    settings = load_settings()
    if not settings.llm_api_key:
        print(f"{settings.llm_key_env} is not set; set {settings.llm_key_env} to run the eval.",
              file=sys.stderr)
        return 1

    golden_path = Path(argv[1]) if len(argv) > 1 else Path("eval/golden.jsonl")
    try:
        store = Store(settings.database_url)
        problems = preflight(golden_path, store)
    except psycopg.Error as exc:
        problems = [_db_problem(exc)]
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        raise SystemExit(1)

    items = load_golden(golden_path)
    file_counts = {
        Path(item["file"]).name: store.count_for_file(Path(item["file"]).name)
        for item in items
    }
    rerank_enabled = getattr(settings, "rerank_enabled", False)
    rerank_model = (
        getattr(settings, "rerank_model", None) or DEFAULT_RERANK_MODEL
        if rerank_enabled
        else None
    )
    reranker = Reranker(rerank_model) if rerank_enabled else None
    candidates = getattr(settings, "rerank_candidates", None) if rerank_enabled else None
    hybrid = getattr(settings, "hybrid_enabled", False)
    results = run_eval(golden_path, store, Embedder(),
                       make_client(settings),
                       settings.model, k=settings.top_k, reranker=reranker,
                       candidates=candidates, hybrid=hybrid)
    Path("eval/report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    Path("eval/report.md").write_text(
        render_markdown(results, settings.model, file_counts,
                        reranker_model=rerank_model, hybrid=hybrid),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in results.items() if k != "per_question"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
