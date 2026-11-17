"""Optional local cross-encoder reranking of retrieved passages."""

from __future__ import annotations

from dataclasses import replace

from fastembed.rerank.cross_encoder import TextCrossEncoder

from ragchat.config import DEFAULT_RERANK_MODEL
from ragchat.store import Hit


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, encoder=None) -> None:
        self._model_name = model_name
        self._model = encoder

    def _load(self):
        if self._model is None:
            self._model = TextCrossEncoder(model_name=self._model_name)
        return self._model

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        scores = list(self._load().rerank(query, [hit.text for hit in hits]))
        ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
        return [replace(hit, score=float(score)) for hit, score in ranked[:top_k]]
