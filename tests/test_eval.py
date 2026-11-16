# tests/test_eval.py
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from ragchat.eval.run_eval import (
    JUDGE_SYSTEM,
    _judge,
    main,
    preflight,
    render_markdown,
    run_eval,
)
from ragchat.eval.validate_golden import EXPECTED_ITEMS, validate_golden
from ragchat.store import Hit

FIXTURES = Path(__file__).parent / "fixtures"


def _write_golden(tmp_path, items, name="g.jsonl"):
    golden = tmp_path / name
    golden.write_text(
        "".join(json.dumps(item) + "\n" for item in items), encoding="utf-8"
    )
    return golden


def _fixture_item(identifier="q1", page=2):
    return {"id": identifier, "question": "?", "file": "sample.pdf",
            "page": page, "evidence_phrase": "authentication"}


def _fixture_items(count):
    return [_fixture_item(f"q{i:02d}") for i in range(1, count + 1)]


def _item(identifier="q1", question="How long?", file="corpus/a.pdf", page=2,
          phrase="eight characters"):
    return {"id": identifier, "question": question, "file": file,
            "page": page, "evidence_phrase": phrase}


def _hit(file="corpus/a.pdf", page=2, text="eight characters", score=0.9):
    return Hit(file=file, page=page, chunk_index=0, text=text, score=score)


# --- validator -----------------------------------------------------------------

def test_validator_flags_a_wrong_page(tmp_path):
    golden = _write_golden(tmp_path, [_fixture_item(page=1)])
    problems = validate_golden(golden, FIXTURES, expected_items=1)
    assert problems and "page 1" in problems[0]


def test_validator_accepts_a_correct_entry(tmp_path):
    golden = _write_golden(tmp_path, [_fixture_item()])
    assert validate_golden(golden, FIXTURES, expected_items=1) == []


def test_validator_rejects_an_empty_file(tmp_path):
    golden = _write_golden(tmp_path, [])

    problems = validate_golden(golden, FIXTURES)

    assert len(problems) == 1
    assert f"expected {EXPECTED_ITEMS} items, found 0" in problems[0]


def test_validator_rejects_a_short_golden_set(tmp_path):
    golden = _write_golden(tmp_path, _fixture_items(EXPECTED_ITEMS - 1))

    problems = validate_golden(golden, FIXTURES)

    assert problems == [f"expected {EXPECTED_ITEMS} items, found {EXPECTED_ITEMS - 1}"]


def test_validator_flags_a_duplicate_id(tmp_path):
    golden = _write_golden(tmp_path, [_fixture_item("q1"), _fixture_item("q1")])

    problems = validate_golden(golden, FIXTURES, expected_items=2)

    assert problems == ["q1: duplicate id"]


def test_validator_reports_malformed_json_without_traceback(tmp_path):
    golden = tmp_path / "g.jsonl"
    golden.write_text(
        json.dumps(_fixture_item()) + "\n{not json\n", encoding="utf-8"
    )

    problems = validate_golden(golden, FIXTURES, expected_items=1)

    assert any("malformed JSON" in problem for problem in problems)


def test_validator_reports_a_missing_key_without_traceback(tmp_path):
    item = _fixture_item()
    del item["page"]
    golden = _write_golden(tmp_path, [item])
    empty_corpus = tmp_path / "corpus"
    empty_corpus.mkdir()

    problems = validate_golden(golden, empty_corpus, expected_items=1)

    assert problems == ["q1: missing required field(s): page"]


def test_validator_reports_a_wrong_type_without_traceback(tmp_path):
    item = _fixture_item()
    item["page"] = "two"
    golden = _write_golden(tmp_path, [item])
    empty_corpus = tmp_path / "corpus"
    empty_corpus.mkdir()

    problems = validate_golden(golden, empty_corpus, expected_items=1)

    assert problems == ["q1: wrong type: page must be an integer"]


def test_validator_flags_an_unrepresented_corpus_pdf(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shutil.copy(FIXTURES / "sample.pdf", corpus / "a.pdf")
    shutil.copy(FIXTURES / "sample.pdf", corpus / "b.pdf")
    item = {"id": "q1", "question": "?", "file": "corpus/a.pdf",
            "page": 2, "evidence_phrase": "authentication"}
    golden = _write_golden(tmp_path, [item])

    problems = validate_golden(golden, corpus, expected_items=1)

    assert problems == ["corpus file not represented in golden set: b.pdf"]


# --- fakes ---------------------------------------------------------------------

class _FakeEmbedder:
    def embed(self, texts):
        return [[0.0] * 384 for _ in texts]


class _FakeStore:
    """One flat hit list reused for every search, or a list of per-search lists.

    The default k is deliberately small so a runner that forgets to pass k fails.
    """

    def __init__(self, hits):
        if hits and isinstance(hits[0], Hit):
            self._batches = [list(hits)]
        else:
            self._batches = [list(batch) for batch in hits]
        self._next = 0
        self.k_values = []

    def search(self, vector, k=1):
        self.k_values.append(k)
        if self._next < len(self._batches):
            batch = self._batches[self._next]
            self._next += 1
        else:
            batch = self._batches[-1] if self._batches else []
        return batch[:k]


class _FakeClient:
    """Answers with queued text and grades with queued judge replies."""

    def __init__(self, answers=("The minimum is eight characters [1].",),
                 judges=("SCORE 3 every claim is supported",)):
        self.answers = list(answers)
        self.judges = list(judges)
        self.messages = self

    def create(self, **kwargs):
        if kwargs.get("system") == JUDGE_SYSTEM:
            text = self.judges.pop(0)
        else:
            text = self.answers.pop(0)
        block = type("Block", (), {"type": "text", "text": text})()
        return type("Msg", (), {"content": [block]})()


# --- judge parser --------------------------------------------------------------

@pytest.mark.parametrize(("reply", "expected"), [
    ("SCORE 2. Mostly supported.", 2 / 3),
    ("**SCORE 3** Every claim is supported.", 1.0),
    ("3 sources were used; SCORE 2 mostly supported.", 2 / 3),
    ("`SCORE 1` partially supported.", 1 / 3),
])
def test_judge_reads_the_score_label(reply, expected):
    client = _FakeClient(judges=[reply])

    assert _judge(client, "m", "q", "a", "s") == pytest.approx(expected)


def test_judge_keeps_a_genuine_zero():
    client = _FakeClient(judges=["SCORE 0 contradicts the sources."])

    assert _judge(client, "m", "q", "a", "s") == 0.0


def test_judge_returns_none_when_no_score_is_present():
    client = _FakeClient(judges=["I cannot grade this answer."])

    assert _judge(client, "m", "q", "a", "s") is None


# --- metrics -------------------------------------------------------------------

def test_run_eval_averages_over_every_question(tmp_path):
    golden = _write_golden(tmp_path, [_item("q1"), _item("q2")])
    store = _FakeStore([
        [_hit("corpus/a.pdf", 2)],
        [_hit("corpus/a.pdf", 9, text="unrelated", score=0.4)],
    ])
    client = _FakeClient(answers=["Answer [1].", "Answer [1]."],
                         judges=["SCORE 3 supported", "SCORE 3 supported"])

    results = run_eval(golden, store, _FakeEmbedder(), client, "m", k=5)

    assert results["n"] == 2
    assert results["hit_at_k"] == 0.5
    assert results["mrr"] == 0.5
    assert results["grounding_rate"] == 0.5
    assert results["faithfulness"] == 1.0
    assert results["judge_unparsed"] == 0
    per_question = results["per_question"]
    assert [q["id"] for q in per_question] == ["q1", "q2"]
    assert [q["hit"] for q in per_question] == [True, False]
    assert [q["rank"] for q in per_question] == [1, None]
    assert [q["cited_gold_page"] for q in per_question] == [True, False]


def test_run_eval_scores_a_hit_at_rank_one(tmp_path):
    golden = _write_golden(tmp_path, [_item(page=2)])

    results = run_eval(golden, _FakeStore([_hit("corpus/a.pdf", 2)]),
                       _FakeEmbedder(), _FakeClient(), "m", k=5)

    assert results["n"] == 1
    assert results["hit_at_k"] == 1.0
    assert results["mrr"] == 1.0
    assert results["grounding_rate"] == 1.0
    assert results["faithfulness"] == 1.0


def test_run_eval_scores_a_miss_at_zero(tmp_path):
    """The retrieved page is not the gold page, so every retrieval metric must be 0."""
    golden = _write_golden(tmp_path, [_item(page=2)])
    hits = [_hit("corpus/a.pdf", 9, text="unrelated", score=0.4)]

    results = run_eval(golden, _FakeStore(hits), _FakeEmbedder(), _FakeClient(), "m", k=5)

    assert results["hit_at_k"] == 0.0
    assert results["mrr"] == 0.0
    assert results["grounding_rate"] == 0.0


def test_run_eval_mrr_uses_the_rank(tmp_path):
    golden = _write_golden(tmp_path, [_item(page=2)])
    hits = [
        _hit("corpus/a.pdf", 9, text="unrelated"),
        _hit("corpus/a.pdf", 2, text="eight characters", score=0.8),
    ]
    store = _FakeStore(hits)

    results = run_eval(golden, store, _FakeEmbedder(), _FakeClient(), "m", k=5)

    assert results["hit_at_k"] == 1.0
    assert results["mrr"] == 0.5
    assert store.k_values == [5]


def test_run_eval_only_sees_the_top_k(tmp_path):
    golden = _write_golden(tmp_path, [_item(page=2)])
    hits = [_hit("corpus/a.pdf", page) for page in (9, 10, 11, 12, 13, 2)]
    store = _FakeStore(hits)

    results = run_eval(golden, store, _FakeEmbedder(), _FakeClient(), "m", k=5)

    assert results["hit_at_k"] == 0.0
    assert results["mrr"] == 0.0
    assert store.k_values == [5]


def test_run_eval_does_not_count_a_hit_on_a_suffix_named_file(tmp_path):
    """old-nist-sp-800-63b.pdf must not satisfy a gold item for nist-sp-800-63b.pdf."""
    golden = _write_golden(tmp_path, [_item(file="corpus/nist-sp-800-63b.pdf", page=2)])
    hits = [_hit("corpus/old-nist-sp-800-63b.pdf", 2)]

    results = run_eval(golden, _FakeStore(hits), _FakeEmbedder(), _FakeClient(), "m", k=5)

    assert results["hit_at_k"] == 0.0
    assert results["mrr"] == 0.0
    assert results["grounding_rate"] == 0.0


def test_run_eval_ranks_the_correct_file_not_just_the_page(tmp_path):
    """Both files have page 2; the wrong-file hit at rank 1 must not count as gold."""
    golden = _write_golden(tmp_path, [_item(file="corpus/nist-sp-800-63b.pdf", page=2)])
    hits = [
        _hit("corpus/old-nist-sp-800-63b.pdf", 2),
        _hit("corpus/nist-sp-800-63b.pdf", 2),
    ]

    results = run_eval(golden, _FakeStore(hits), _FakeEmbedder(), _FakeClient(), "m", k=5)

    assert results["hit_at_k"] == 1.0
    assert results["mrr"] == 0.5


def test_run_eval_grounding_requires_the_cited_gold_page(tmp_path):
    golden = _write_golden(tmp_path, [_item(page=2)])
    hits = [
        _hit("corpus/a.pdf", 2),
        _hit("corpus/a.pdf", 9, text="wrong page", score=0.8),
    ]
    client = _FakeClient(answers=["Supported by source [2]."])

    results = run_eval(golden, _FakeStore(hits), _FakeEmbedder(), client, "m", k=5)

    assert results["hit_at_k"] == 1.0
    assert results["mrr"] == 1.0
    assert results["grounding_rate"] == 0.0


def test_run_eval_faithfulness_uses_the_judge_score(tmp_path):
    golden = _write_golden(tmp_path, [_item(page=2)])
    client = _FakeClient(judges=["SCORE 1 partially supported"])

    results = run_eval(golden, _FakeStore([_hit("corpus/a.pdf", 2)]),
                       _FakeEmbedder(), client, "m", k=5)

    assert results["faithfulness"] == 0.333
    assert results["judge_unparsed"] == 0


def test_run_eval_excludes_unparsed_judge_scores(tmp_path):
    golden = _write_golden(tmp_path, [_item("q1"), _item("q2")])
    store = _FakeStore([[_hit("corpus/a.pdf", 2)], [_hit("corpus/a.pdf", 2)]])
    client = _FakeClient(
        answers=["Answer [1].", "Answer [1]."],
        judges=["SCORE 3 supported", "I cannot grade this answer."],
    )

    results = run_eval(golden, store, _FakeEmbedder(), client, "m", k=5)

    assert results["faithfulness"] == 1.0
    assert results["judge_unparsed"] == 1
    assert results["per_question"][1]["faithfulness"] is None


# --- markdown ------------------------------------------------------------------

def test_render_markdown_reports_headline_values_and_unparsed_count(tmp_path):
    golden = _write_golden(tmp_path, [_item("q1"), _item("q2")])
    store = _FakeStore([
        [_hit("corpus/a.pdf", 2)],
        [_hit("corpus/a.pdf", 9, text="unrelated", score=0.4)],
    ])
    results = run_eval(golden, store, _FakeEmbedder(),
                       _FakeClient(answers=["Answer [1].", "Answer [1]."],
                                   judges=["SCORE 3 supported", "SCORE 3 supported"]),
                       "m", k=5)

    markdown = render_markdown(results, "m", {"a.pdf": 40, "b.pdf": 60})

    assert "Retrieval hit@5" in markdown
    assert "50.0%" in markdown
    assert "| MRR | **0.500** |" in markdown
    assert "Citation points at the gold page" in markdown
    assert "| Faithfulness (LLM judge, 0–1) | **1.00** |" in markdown
    assert "100 chunks across the golden files (a.pdf: 40; b.pdf: 60)" in markdown
    assert "| Judge replies unparsed | 0 |" in markdown


def test_render_markdown_surfaces_unparsed_judge_replies(tmp_path):
    golden = _write_golden(tmp_path, [_item(page=2)])
    client = _FakeClient(judges=["no verdict"])
    results = run_eval(golden, _FakeStore([_hit("corpus/a.pdf", 2)]),
                       _FakeEmbedder(), client, "m", k=5)

    markdown = render_markdown(results, "m", {"a.pdf": 1})

    assert "| Judge replies unparsed | 1 |" in markdown
    assert "could not be parsed" in markdown
    assert "| q1 | yes | 1 | yes | - |" in markdown


# --- preflight / main ----------------------------------------------------------

class _PreflightStore:
    def __init__(self, total, counts=None):
        self.total = total
        self.counts = counts or {}

    def count(self):
        return self.total

    def count_for_file(self, name):
        return self.counts.get(name, 0)


def _full_golden(tmp_path, count=EXPECTED_ITEMS):
    items = [
        _item(f"q{i:02d}", file="corpus/a.pdf" if i % 2 else "corpus/b.pdf")
        for i in range(1, count + 1)
    ]
    return _write_golden(tmp_path, items)


def test_preflight_rejects_an_empty_store(tmp_path):
    golden = _full_golden(tmp_path)

    problems = preflight(golden, _PreflightStore(0))

    assert problems == ["store is empty; ingest the corpus before running the eval"]


def test_preflight_rejects_a_short_golden_set(tmp_path):
    golden = _full_golden(tmp_path, count=EXPECTED_ITEMS - 1)
    store = _PreflightStore(100, {"a.pdf": 50, "b.pdf": 50})

    problems = preflight(golden, store)

    assert any(
        f"{EXPECTED_ITEMS - 1} item(s), expected {EXPECTED_ITEMS}" in problem
        for problem in problems
    )


def test_preflight_rejects_a_golden_file_with_no_chunks(tmp_path):
    golden = _full_golden(tmp_path)
    store = _PreflightStore(100, {"a.pdf": 100, "b.pdf": 0})

    problems = preflight(golden, store)

    assert any("no chunks in the store for b.pdf" in problem for problem in problems)


def test_main_fails_fast_without_an_api_key(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    settings = SimpleNamespace(anthropic_api_key="", database_url="x",
                               model="m", top_k=5)
    monkeypatch.setattr("ragchat.config.load_settings", lambda: settings)

    assert main(["run_eval"]) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "eval/report.json").exists()
    assert not (tmp_path / "eval/report.md").exists()


def test_main_writes_no_report_when_preflight_fails(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval/golden.jsonl").write_text("", encoding="utf-8")
    settings = SimpleNamespace(anthropic_api_key="key", database_url="x",
                               model="m", top_k=5)
    monkeypatch.setattr("ragchat.config.load_settings", lambda: settings)
    monkeypatch.setattr("ragchat.eval.run_eval.Store", lambda url: _PreflightStore(0))

    assert main(["run_eval"]) == 1
    assert "Preflight failed; no report written." in capsys.readouterr().err
    assert not (tmp_path / "eval/report.json").exists()
    assert not (tmp_path / "eval/report.md").exists()
