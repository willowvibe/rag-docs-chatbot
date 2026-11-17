# tests/test_answer.py
import pytest

from ragchat.answer import answer, build_messages, parse_citations
from ragchat.store import Hit

HITS = [
    Hit(file="a.pdf", page=4, chunk_index=0, text="Passwords SHALL be at least 8 characters.", score=0.8),
    Hit(file="b.pdf", page=11, chunk_index=3, text="Rate limiting SHALL be applied.", score=0.7),
]


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = type("Block", (), {"type": "text", "text": self.reply})()
        return type("Msg", (), {"content": [block]})()


class FlakyClient:
    def __init__(self, failures, reply):
        self.failures = list(failures)
        self.reply = reply
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        block = type("Block", (), {"type": "text", "text": self.reply})()
        return type("Msg", (), {"content": [block]})()


def test_build_messages_numbers_sources_and_names_file_and_page():
    messages = build_messages("How long must passwords be?", HITS)
    user = messages[-1]["content"]
    assert "[1] a.pdf, page 4" in user
    assert "[2] b.pdf, page 11" in user
    assert "How long must passwords be?" in user
    assert messages[0]["role"] == "system"
    assert "only" in messages[0]["content"].lower()


def test_parse_citations_extracts_unique_markers_in_order():
    text = "At least 8 characters [1], and rate limiting applies [2]. See also [1]."
    cits = parse_citations(text, HITS)
    assert [(c.file, c.page) for c in cits] == [("a.pdf", 4), ("b.pdf", 11)]
    assert cits[0].snippet.startswith("Passwords SHALL")


def test_parse_citations_ignores_out_of_range_markers():
    assert parse_citations("Claim [7].", HITS) == []


def test_parse_citations_resolves_the_marker_number_not_its_position():
    """Pins the contract: marker [2] means the second source, not "the first marker found"."""
    cits = parse_citations("Only [2].", HITS)
    assert [(c.file, c.page) for c in cits] == [("b.pdf", 11)]
    assert cits[0].marker == 2


def test_parse_citations_keeps_first_appearance_order():
    cits = parse_citations("[2] then [1]", HITS)
    assert [(c.file, c.page) for c in cits] == [
        ("b.pdf", 11),
        ("a.pdf", 4),
    ]
    assert [c.marker for c in cits] == [2, 1]


def test_answer_returns_text_and_citations():
    client = FakeClient("Passwords must be at least 8 characters [1].")
    result = answer("How long?", HITS, client=client)
    assert result.text.endswith("[1].")
    assert result.citations[0].file == "a.pdf"
    assert len(result.used_hits) == 2
    assert client.calls[0]["model"] == "claude-sonnet-5-5"
    assert client.calls[0]["max_tokens"] == 700


def test_answer_normalizes_bracketed_marker_with_source():
    client = FakeClient("Passwords must be at least 8 characters 【1†corpus/a.pdf, page 4】.")
    result = answer("How long?", HITS, client=client)
    assert result.text == "Passwords must be at least 8 characters [1]."
    assert [(c.file, c.page) for c in result.citations] == [("a.pdf", 4)]


def test_answer_normalizes_bracketed_marker_with_page_only():
    client = FakeClient("Rate limiting applies 【2†page 11】.")
    result = answer("Rate limiting?", HITS, client=client)
    assert result.text == "Rate limiting applies [2]."
    assert [(c.file, c.page) for c in result.citations] == [("b.pdf", 11)]


def test_answer_normalizes_ascii_marker_with_source_and_page():
    client = FakeClient("Too many requests, HTTP 429 [2, api.pdf, page 11].")
    result = answer("Rate limiting?", HITS, client=client)
    assert result.text == "Too many requests, HTTP 429 [2]."
    assert [(c.file, c.page) for c in result.citations] == [("b.pdf", 11)]


def test_answer_normalizes_ascii_marker_with_dagger():
    client = FakeClient("Rate limiting applies [2†page 11].")
    result = answer("Rate limiting?", HITS, client=client)
    assert result.text == "Rate limiting applies [2]."
    assert [(c.file, c.page) for c in result.citations] == [("b.pdf", 11)]


def test_answer_normalizes_ascii_marker_with_space_dash():
    client = FakeClient("Rate limiting applies [2 - api.pdf].")
    result = answer("Rate limiting?", HITS, client=client)
    assert result.text == "Rate limiting applies [2]."
    assert [(c.file, c.page) for c in result.citations] == [("b.pdf", 11)]


def test_answer_normalizes_ascii_multi_number_marker_to_separate_citations():
    client = FakeClient("Both apply [1, 2].")
    result = answer("What applies?", HITS, client=client)
    assert result.text == "Both apply [1][2]."
    assert [(c.file, c.page) for c in result.citations] == [
        ("a.pdf", 4),
        ("b.pdf", 11),
    ]


def test_answer_resolves_source_only_marker_with_file_and_page():
    client = FakeClient("Rate limiting applies [b.pdf, page 11].")
    result = answer("Rate limiting?", HITS, client=client)
    assert result.text == "Rate limiting applies [2]."
    assert [(c.file, c.page) for c in result.citations] == [("b.pdf", 11)]


def test_answer_resolves_source_only_marker_with_path_prefix():
    client = FakeClient("Rate limiting applies [corpus/b.pdf, page 11].")
    result = answer("Rate limiting?", HITS, client=client)
    assert result.text == "Rate limiting applies [2]."
    assert [(c.file, c.page) for c in result.citations] == [("b.pdf", 11)]


def test_answer_resolves_source_only_marker_with_p_abbreviation():
    client = FakeClient("Rate limiting applies [b.pdf p. 11].")
    result = answer("Rate limiting?", HITS, client=client)
    assert result.text == "Rate limiting applies [2]."
    assert [(c.file, c.page) for c in result.citations] == [("b.pdf", 11)]


def test_answer_leaves_unknown_source_only_marker_unchanged():
    client = FakeClient("See [missing.pdf, page 11] and [b.pdf, page 4].")
    result = answer("Anything?", HITS, client=client)
    assert result.text == "See [missing.pdf, page 11] and [b.pdf, page 4]."
    assert result.citations == []


def test_answer_leaves_non_numeric_brackets_unchanged():
    client = FakeClient("See [see note] for details.")
    result = answer("Anything?", HITS, client=client)
    assert result.text == "See [see note] for details."
    assert result.citations == []


def test_answer_treats_missing_markers_as_uncited():
    client = FakeClient("I don't know based on the provided documents.")
    result = answer("Unrelated?", HITS, client=client)
    assert result.citations == []
    assert result.text == "I don't know based on the provided documents."


def test_answer_with_no_hits_does_not_call_the_model():
    client = FakeClient("should not be used")
    result = answer("anything", [], client=client)
    assert client.calls == []
    assert result.text == "I don't know based on the provided documents."


def test_answer_retries_with_the_fallback_model_when_the_primary_fails(caplog):
    client = FlakyClient([RuntimeError("primary boom")], "Fallback answer [2].")
    result = answer(
        "Rate limiting?",
        HITS,
        client=client,
        model="primary-model",
        fallback_model="backup-model",
    )

    assert result.text == "Fallback answer [2]."
    assert [(c.file, c.page) for c in result.citations] == [("b.pdf", 11)]
    assert [call["model"] for call in client.calls] == ["primary-model", "backup-model"]
    assert "primary-model" in caplog.text
    assert "backup-model" in caplog.text
    assert "RuntimeError" in caplog.text


def test_answer_retries_with_the_fallback_model_when_the_completion_is_empty(caplog):
    client = FlakyClient(
        [
            RuntimeError(
                "empty completion from primary-model (done_reason=length)"
            )
        ],
        "Fallback answer [2].",
    )
    result = answer(
        "Rate limiting?",
        HITS,
        client=client,
        model="primary-model",
        fallback_model="backup-model",
    )

    assert result.text == "Fallback answer [2]."
    assert [call["model"] for call in client.calls] == ["primary-model", "backup-model"]
    # The warning names the exception type only, never its message (no secret leakage).
    assert "RuntimeError" in caplog.text
    assert "backup-model" in caplog.text


def test_answer_raises_when_the_primary_and_fallback_both_fail():
    client = FlakyClient(
        [RuntimeError("primary boom"), RuntimeError("fallback boom")],
        "unused",
    )
    with pytest.raises(RuntimeError, match="fallback boom"):
        answer(
            "Rate limiting?",
            HITS,
            client=client,
            model="primary-model",
            fallback_model="backup-model",
        )

    assert [call["model"] for call in client.calls] == ["primary-model", "backup-model"]


def test_answer_without_a_fallback_does_not_retry():
    client = FlakyClient([RuntimeError("primary boom")], "unused")
    with pytest.raises(RuntimeError, match="primary boom"):
        answer("Rate limiting?", HITS, client=client, model="primary-model")

    assert [call["model"] for call in client.calls] == ["primary-model"]


def test_answer_does_not_retry_when_the_fallback_equals_the_primary():
    client = FlakyClient([RuntimeError("primary boom")], "unused")
    with pytest.raises(RuntimeError, match="primary boom"):
        answer(
            "Rate limiting?",
            HITS,
            client=client,
            model="same-model",
            fallback_model="same-model",
        )

    assert [call["model"] for call in client.calls] == ["same-model"]
