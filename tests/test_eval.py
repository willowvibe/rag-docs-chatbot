# tests/test_eval.py
import json
from pathlib import Path

from ragchat.eval.run_eval import JUDGE_SYSTEM, run_eval
from ragchat.eval.validate_golden import validate_golden
from ragchat.store import Hit


def test_validator_flags_a_wrong_page(tmp_path):
    golden = tmp_path / "g.jsonl"
    golden.write_text(json.dumps({
        "id": "q1", "question": "?", "file": "sample.pdf", "page": 1,
        "evidence_phrase": "authentication",
    }) + "\n", encoding="utf-8")
    fixtures = Path(__file__).parent / "fixtures"
    problems = validate_golden(golden, fixtures)
    assert problems and "page 1" in problems[0]


def test_validator_accepts_a_correct_entry(tmp_path):
    golden = tmp_path / "g.jsonl"
    golden.write_text(json.dumps({
        "id": "q1", "question": "?", "file": "sample.pdf", "page": 2,
        "evidence_phrase": "authentication",
    }) + "\n", encoding="utf-8")
    assert validate_golden(golden, Path(__file__).parent / "fixtures") == []


class _FakeEmbedder:
    def embed(self, texts):
        return [[0.0] * 384 for _ in texts]


class _FakeStore:
    def __init__(self, hits):
        self.hits = hits

    def search(self, vector, k=5):
        return self.hits[:k]


class _FakeClient:
    """One model for both roles: scores the judge prompt, cites [1] for the answer prompt."""

    def __init__(self):
        self.messages = self

    def create(self, **kwargs):
        if kwargs.get("system") == JUDGE_SYSTEM:
            text = "SCORE 3 every claim is supported"
        else:
            text = "The minimum is eight characters [1]."
        block = type("Block", (), {"type": "text", "text": text})()
        return type("Msg", (), {"content": [block]})()


def _one_question_golden(tmp_path, page):
    golden = tmp_path / "g.jsonl"
    golden.write_text(
        json.dumps({"id": "q1", "question": "How long?", "file": "corpus/a.pdf",
                    "page": page, "evidence_phrase": "eight characters"}) + "\n",
        encoding="utf-8",
    )
    return golden


def test_run_eval_scores_a_hit_at_rank_one(tmp_path):
    golden = _one_question_golden(tmp_path, page=2)
    hits = [Hit(file="corpus/a.pdf", page=2, chunk_index=0, text="eight characters", score=0.9)]

    results = run_eval(golden, _FakeStore(hits), _FakeEmbedder(), _FakeClient(), "m", k=5)

    assert results["n"] == 1
    assert results["hit_at_k"] == 1.0
    assert results["mrr"] == 1.0
    assert results["grounding_rate"] == 1.0
    assert results["faithfulness"] == 1.0


def test_run_eval_scores_a_miss_at_zero(tmp_path):
    """The retrieved page is not the gold page, so every retrieval metric must be 0."""
    golden = _one_question_golden(tmp_path, page=2)
    hits = [Hit(file="corpus/a.pdf", page=9, chunk_index=0, text="unrelated", score=0.4)]

    results = run_eval(golden, _FakeStore(hits), _FakeEmbedder(), _FakeClient(), "m", k=5)

    assert results["hit_at_k"] == 0.0
    assert results["mrr"] == 0.0
    assert results["grounding_rate"] == 0.0


def test_run_eval_mrr_uses_the_rank(tmp_path):
    golden = _one_question_golden(tmp_path, page=2)
    hits = [
        Hit(file="corpus/a.pdf", page=9, chunk_index=0, text="unrelated", score=0.9),
        Hit(file="corpus/a.pdf", page=2, chunk_index=1, text="eight characters", score=0.8),
    ]

    results = run_eval(golden, _FakeStore(hits), _FakeEmbedder(), _FakeClient(), "m", k=5)

    assert results["hit_at_k"] == 1.0
    assert results["mrr"] == 0.5
