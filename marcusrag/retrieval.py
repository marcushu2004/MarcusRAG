"""Retrieval strategies over any `Store` implementation.

Three modes, all behind one `invoke(question)` entry point:

- `similarity`: plain vector search;
- `mmr`: vector search + maximal marginal relevance (diversity);
- `hybrid`  : vector search **and** BM25, fused with reciprocal rank fusion,
  then optionally re-ranked by a cross-encoder.

The hybrid path deliberately fuses on *document ids* rather than on scores:
cosine similarity and BM25 are on incomparable scales, and min-max normalising
them makes the result depend on the candidate set. RRF only needs rank order.

The BM25 half is built lazily from `Store.all_items()` and rebuilt whenever the
store size changes, so `index` → `ask` never uses a stale lexical index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .bm25 import BM25Index, reciprocal_rank_fusion
from .embeddings import Embedder
from .rerank import Reranker
from .vectorstore import Store, _cosine


@dataclass
class Retrieved:
    """A single retrieved chunk."""

    doc_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    detail: dict[str, float] = field(default_factory=dict)

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
    """Vector + lexical retrieval with optional MMR and re-ranking.

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
        search_type: str = "hybrid",
        mmr_lambda: float = 0.5,
        reranker: Reranker | None = None,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = max(top_k, 1)
        self.fetch_k = max(fetch_k, self.top_k)
        self.search_type = (search_type or "hybrid").lower()
        self.mmr_lambda = min(max(mmr_lambda, 0.0), 1.0)
        self.reranker = reranker
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = max(rrf_k, 1)
        self._bm25: BM25Index | None = None
        self._bm25_size = -1

    # ------------------------------------------------------------- internals
    def _lexical_index(self) -> BM25Index:
        size = self.store.count()
        if self._bm25 is None or self._bm25_size != size:
            items = [(doc_id, text) for doc_id, text, _ in self.store.all_items()]
            self._bm25 = BM25Index(items)
            self._bm25_size = size
        return self._bm25

    def refresh(self) -> None:
        """Drop cached lexical state (call after re-indexing in-process)."""
        self._bm25 = None
        self._bm25_size = -1

    # ---------------------------------------------------------------- public
    def invoke(self, question: str) -> list[Retrieved]:
        if not question.strip():
            return []
        if self.search_type == "hybrid":
            return self._hybrid(question)
        return self._vector_only(question)

    # ------------------------------------------------------------ strategies
    def _vector_only(self, question: str) -> list[Retrieved]:
        query_vector = self.embedder.embed_query(question)
        hits = self.store.query(query_vector, self.fetch_k)
        candidates = [Retrieved(*hit) for hit in hits]
        if not candidates:
            return []
        if self.search_type != "mmr" or len(candidates) <= 1:
            return candidates[: self.top_k]
        return self._mmr(query_vector, candidates)

    def _hybrid(self, question: str) -> list[Retrieved]:
        query_vector = self.embedder.embed_query(question)
        vector_hits = self.store.query(query_vector, self.fetch_k)
        lexical_hits = self._lexical_index().search(question, self.fetch_k)

        vector_ids = [hit[0] for hit in vector_hits]
        lexical_ids = [doc_id for doc_id, _ in lexical_hits]
        if not vector_ids and not lexical_ids:
            return []

        fused = reciprocal_rank_fusion(
            [vector_ids, lexical_ids],
            k=self.rrf_k,
            weights=[self.vector_weight, self.bm25_weight],
        )

        pool: dict[str, tuple[str, str, dict[str, Any], float]] = {hit[0]: hit for hit in vector_hits}
        missing = [doc_id for doc_id, _ in fused if doc_id not in pool]
        if missing:
            for hit in self.store.get(missing):
                pool[hit[0]] = hit

        vector_scores = {hit[0]: hit[3] for hit in vector_hits}
        lexical_scores = dict(lexical_hits)

        candidates: list[Retrieved] = []
        for doc_id, rrf_score in fused:
            hit = pool.get(doc_id)
            if hit is None:  # id vanished between the two lookups
                continue
            candidates.append(
                Retrieved(
                    doc_id=doc_id,
                    text=hit[1],
                    metadata=hit[2],
                    score=rrf_score,
                    detail={
                        "rrf": rrf_score,
                        "vector": vector_scores.get(doc_id, 0.0),
                        "bm25": lexical_scores.get(doc_id, 0.0),
                    },
                )
            )

        if not candidates:
            return []

        if self.reranker is not None:
            candidates = self._rerank(question, candidates)

        return candidates[: self.top_k]

    def _rerank(self, question: str, candidates: list[Retrieved]) -> list[Retrieved]:
        assert self.reranker is not None
        scores = self.reranker.score(
            question, [c.text for c in candidates], base=[c.score for c in candidates]
        )
        if len(scores) != len(candidates):
            raise RuntimeError(
                f"reranker {self.reranker.name} returned {len(scores)} scores for {len(candidates)} candidates"
            )
        rescored: list[Retrieved] = []
        for candidate, score in zip(candidates, scores):
            detail = dict(candidate.detail)
            detail["base_score"] = round(float(candidate.score), 6)
            detail["rerank"] = float(score)
            rescored.append(Retrieved(candidate.doc_id, candidate.text, candidate.metadata, float(score), detail))
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored

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
    reranker = None
    if getattr(settings, "rerank", "none") not in (None, "", "none", "off"):
        from .rerank import build_reranker

        reranker = build_reranker(
            settings.rerank,
            settings.rerank_model,
            base_weight=getattr(settings, "rerank_base_weight", 0.7),
        )
    return Retriever(
        store,
        embedder,
        top_k=settings.top_k,
        fetch_k=settings.fetch_k,
        search_type=settings.search_type,
        mmr_lambda=settings.mmr_lambda,
        reranker=reranker,
        vector_weight=getattr(settings, "hybrid_vector_weight", 1.0),
        bm25_weight=getattr(settings, "hybrid_bm25_weight", 1.0),
        rrf_k=getattr(settings, "rrf_k", 60),
    )
