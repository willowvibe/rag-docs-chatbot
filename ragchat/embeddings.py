"""Local ONNX embeddings. No API key, no network at query time after the first download."""

from __future__ import annotations

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"


class Embedder:
    dim = 384

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model = TextEmbedding(model_name=model_name)
        self._cache: dict[str, list[float]] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            for text, vector in zip(missing, self._model.embed(missing)):
                self._cache[text] = [float(x) for x in vector]
        # Return copies so callers mutating a vector cannot corrupt the cache.
        return [list(self._cache[t]) for t in texts]

    def clear_cache(self) -> None:
        self._cache.clear()
