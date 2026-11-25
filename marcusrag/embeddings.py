"""Embedding backends.

Two implementations, both dependency-light at import time:

- `HfEmbedder`: sentence-transformers, imported lazily so that `import
  marcusrag` never pulls in torch.
- `HashEmbedder`: a deterministic hashing embedder used for offline smoke tests
  and CI. It is NOT a semantic model — it is only good enough to prove the
  pipeline works end to end without a GPU.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, Sequence


class Embedder(Protocol):
    name: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


class HfEmbedder:
    """sentence-transformers backend with batched encoding."""

    def __init__(self, model_name: str, *, batch_size: int = 32, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
            raise ModuleNotFoundError(
                "sentence-transformers is required for embedder=hf: "
                "pip install sentence-transformers (or use --embedder hash for an offline demo)"
            ) from exc

        self.name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self.dimension = int(self._dimension())
        self.max_seq_length = int(getattr(self._model, "max_seq_length", 0) or 0)

    def _dimension(self) -> int:
        """sentence-transformers 6.0 renamed this accessor; support both."""
        getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        return int(getter())

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, row)) for row in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HashEmbedder:
    """Deterministic character n-gram hashing embedder (offline fallback)."""

    def __init__(self, dimension: int = 256, ngram: int = 3) -> None:
        self.name = f"hash-{dimension}"
        self.dimension = dimension
        self.ngram = ngram

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        normalized = " ".join(text.lower().split())
        if not normalized:
            return vector
        grams = (
            [normalized[i : i + self.ngram] for i in range(max(len(normalized) - self.ngram + 1, 1))]
            if len(normalized) >= self.ngram
            else [normalized]
        )
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimension
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[index] += sign
        return _l2_normalize(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embedder(kind: str, model_name: str, *, batch_size: int = 32) -> Embedder:
    kind = (kind or "hf").lower()
    if kind == "hash":
        return HashEmbedder()
    if kind == "hf":
        return HfEmbedder(model_name, batch_size=batch_size)
    raise ValueError(f"unknown embedder: {kind} (expected 'hf' or 'hash')")
