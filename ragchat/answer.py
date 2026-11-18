# ragchat/answer.py
"""Grounded answering: the model sees only retrieved chunks and must cite them by number."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ragchat.store import Hit

logger = logging.getLogger(__name__)

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
    marker: int = 0


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation]
    used_hits: list[Hit]
    model: str = ""


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    sources = "\n\n".join(
        f"[{i}] {h.file}, page {h.page}\n{h.text}" for i, h in enumerate(hits, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"SOURCES:\n{sources}\n\nQUESTION: {question}"},
    ]


_FULLWIDTH_MARKER = re.compile(r"【\s*(\d{1,2})\s*(?:†[^】]*)?】")
_NUMBER_LIST_MARKER = re.compile(r"\[([ \t]*\d{1,2}(?:[ \t]*,[ \t]*\d{1,2})+[ \t]*)\]")
_DETAILED_MARKER = re.compile(r"\[(\d{1,2})(?:[ \t]*[,†:][ \t]*|[ \t]+-[ \t]*)[^\]\n]*\]")
_SOURCE_ONLY_MARKER = re.compile(
    r"\[\s*(?P<file>(?!\d)[^\[\]\n,]+?)\s*(?:,\s*|\s+)"
    r"(?:page|p\.?)\s*(?P<page>\d{1,4})\s*\]",
    re.IGNORECASE,
)


def normalize_citation_markers(text: str) -> str:
    """Rewrite citation markers with extraneous details to the standard `[n]` form."""
    text = _FULLWIDTH_MARKER.sub(r"[\1]", text)
    text = _NUMBER_LIST_MARKER.sub(
        lambda m: "".join(f"[{n}]" for n in re.findall(r"\d{1,2}", m.group(1))),
        text,
    )
    return _DETAILED_MARKER.sub(r"[\1]", text)


def _basename(path: str) -> str:
    return re.split(r"[\\/]", path.strip())[-1].strip().casefold()


def resolve_source_only_markers(text: str, hits: list[Hit]) -> str:
    """Rewrite `[file, page N]` markers to `[n]` when they uniquely identify a hit."""

    def replace(match: re.Match) -> str:
        file = _basename(match.group("file"))
        page = int(match.group("page"))
        matches = [
            i
            for i, hit in enumerate(hits, start=1)
            if _basename(hit.file) == file and hit.page == page
        ]
        if len(matches) == 1:
            return f"[{matches[0]}]"
        return match.group(0)

    return _SOURCE_ONLY_MARKER.sub(replace, text)


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
        citations.append(Citation(file=hit.file, page=hit.page, snippet=snippet, marker=n))
    return citations


def answer(
    question: str,
    hits: list[Hit],
    client=None,
    model: str = "claude-sonnet-5-5",
    fallbacks: list[tuple[object, str]] | None = None,
) -> Answer:
    if not hits:
        return Answer(text=REFUSAL, citations=[], used_hits=[], model=model)
    if client is None:
        from ragchat.config import load_settings
        from ragchat.llm import make_client, make_provider_client

        settings = load_settings()
        client = make_client(settings)
        if fallbacks is None:
            fallbacks = [
                (make_provider_client(settings, provider), fallback_model)
                for provider, fallback_model in settings.fallback_chain
            ]
    attempts = [(client, model), *(fallbacks or [])]
    for index, (attempt_client, attempt_model) in enumerate(attempts):
        try:
            response = attempt_client.messages.create(
                model=attempt_model,
                max_tokens=700,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=build_messages(question, hits)[1:],
            )
        except Exception as exc:
            # Log the model and exception type only: provider messages can echo
            # request details, and the key must never reach the logs.
            logger.warning(
                "Model %s failed with %s; trying the next model",
                attempt_model,
                type(exc).__name__,
            )
            if index == len(attempts) - 1:
                raise
            continue
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        text = normalize_citation_markers(text)
        text = resolve_source_only_markers(text, hits)
        return Answer(
            text=text,
            citations=parse_citations(text, hits),
            used_hits=hits,
            model=attempt_model,
        )
    raise AssertionError("unreachable")
