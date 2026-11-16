# ragchat/answer.py
"""Grounded answering: the model sees only retrieved chunks and must cite them by number."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragchat.store import Hit

SYSTEM_PROMPT = """You answer questions using ONLY the numbered sources provided.

Rules:
- Cite every factual claim with the source number in square brackets, like [2].
- Use the source's file name and page exactly as given; never invent a page.
- If the sources do not contain the answer, reply exactly:
  I don't know based on the provided documents.
- Do not use outside knowledge, even if you are confident.
- Answer in at most 6 sentences. Quote sparingly, at most one short phrase per source.
"""

REFUSAL = "I don't know based on the provided documents."


@dataclass(frozen=True)
class Citation:
    file: str
    page: int
    snippet: str


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation]
    used_hits: list[Hit]


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    sources = "\n\n".join(
        f"[{i}] {h.file}, page {h.page}\n{h.text}" for i, h in enumerate(hits, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"SOURCES:\n{sources}\n\nQUESTION: {question}"},
    ]


def parse_citations(text: str, hits: list[Hit]) -> list[Citation]:
    seen: list[int] = []
    for raw in re.findall(r"\[(\d{1,2})\]", text):
        n = int(raw)
        if 1 <= n <= len(hits) and n not in seen:
            seen.append(n)
    citations = []
    for n in seen:
        hit = hits[n - 1]
        snippet = hit.text[:300].strip()
        citations.append(Citation(file=hit.file, page=hit.page, snippet=snippet))
    return citations


def answer(
    question: str, hits: list[Hit], client=None, model: str = "claude-sonnet-5-5"
) -> Answer:
    if not hits:
        return Answer(text=REFUSAL, citations=[], used_hits=[])
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=700,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=build_messages(question, hits)[1:],
    )
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    return Answer(text=text, citations=parse_citations(text, hits), used_hits=hits)
