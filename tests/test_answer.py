# tests/test_answer.py
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
    assert [(c.file, c.page) for c in parse_citations("Only [2].", HITS)] == [("b.pdf", 11)]


def test_parse_citations_keeps_first_appearance_order():
    assert [(c.file, c.page) for c in parse_citations("[2] then [1]", HITS)] == [
        ("b.pdf", 11),
        ("a.pdf", 4),
    ]


def test_answer_returns_text_and_citations():
    client = FakeClient("Passwords must be at least 8 characters [1].")
    result = answer("How long?", HITS, client=client)
    assert result.text.endswith("[1].")
    assert result.citations[0].file == "a.pdf"
    assert len(result.used_hits) == 2
    assert client.calls[0]["model"] == "claude-sonnet-5-5"
    assert client.calls[0]["max_tokens"] == 700


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
