"""Retrieval: similarity or MMR over any `Store` implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .embeddings import Embedder
from .vectorstore import Store, _cosine


@dataclass
class Retrieved:
    """A single retrieved chunk."""

    doc_id: str
    text: str
    metadata: dict[str, Any]
    score: float

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))

    @property
    def page(self) -> int | None:
        value = self.metadata.get("page")
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None


class Retriever:
    """Fetches `fetch_k` candidates, then re-ranks with MMR when requested.

    MMR needs candidate vectors. Rather than leaking embeddings out of the store,
    the small candidate set (fetch_k, default 20) is re-encoded here — cheap and
    keeps the store interface minimal.
    """

    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        *,
        top_k: int = 4,
        fetch_k: int = 20,
        search_type: str = "mmr",
        mmr_lambda: float = 0.5,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = max(top_k, 1)
        self.fetch_k = max(fetch_k, self.top_k)
        self.search_type = search_type
        self.mmr_lambda = min(max(mmr_lambda, 0.0), 1.0)

    def invoke(self, question: str) -> list[Retrieved]:
        if not question.strip():
            return []
        query_vector = self.embedder.embed_query(question)
        hits = self.store.query(query_vector, self.fetch_k)
        candidates = [Retrieved(*hit) for hit in hits]
        if not candidates:
            return []
        if self.search_type != "mmr" or len(candidates) <= 1:
            return candidates[: self.top_k]
        return self._mmr(query_vector, candidates)

    def _mmr(self, query_vector: Sequence[float], candidates: list[Retrieved]) -> list[Retrieved]:
        vectors = self.embedder.embed_documents([c.text for c in candidates])
        selected: list[int] = []
        remaining = list(range(len(candidates)))

        while remaining and len(selected) < self.top_k:
            best_index = -1
            best_score = float("-inf")
            for index in remaining:
                relevance = candidates[index].score
                if selected:
                    redundancy = max(_cosine(vectors[index], vectors[j]) for j in selected)
                else:
                    redundancy = 0.0
                score = self.mmr_lambda * relevance - (1.0 - self.mmr_lambda) * redundancy
                if score > best_score:
                    best_score = score
                    best_index = index
            selected.append(best_index)
            remaining.remove(best_index)

        return [candidates[i] for i in selected]


def build_retriever(store: Store, embedder: Embedder, settings) -> Retriever:
    return Retriever(
        store,
        embedder,
        top_k=settings.top_k,
        fetch_k=settings.fetch_k,
        search_type=settings.search_type,
        mmr_lambda=settings.mmr_lambda,
    )
