"""Cross-encoder style re-ranking, with a dependency free fallback.

Two implementations behind one protocol:

- `LexicalReranker`: no model, scores a chunk by how much of the query it
  actually covers (idf-weighted term coverage) plus how tightly the matches are
  clustered. It is not a neural re-ranker, but it is deterministic, free, and
  measurable — which is what an evaluation harness needs.
- `CrossEncoderReranker`: sentence-transformers CrossEncoder (e.g.
  `BAAI/bge-reranker-base`), imported lazily so `import marcusrag` stays cheap.

Both return (chunk_index, score) pairs where the index refers to the input
candidate list, so the retriever keeps ownership of ordering.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Protocol, Sequence

from .bm25 import tokenize


class Reranker(Protocol):
    name: str

    def score(
        self, query: str, candidates: Sequence[str], base: Sequence[float] | None = None
    ) -> list[float]: ...


class LexicalReranker:
    """idf-weighted coverage + proximity, no external model.

    Coverage alone rewards long chunks (they contain more terms by accident), so
    the score also penalises length: a chunk that covers the query in three
    sentences beats one that covers it once inside two thousand words.

    `base_weight` blends the lexical score with the first-stage retrieval score
    (min-max normalised). A pure lexical re-rank ignores the vector channel
    completely and can push a correct hit out of the cut-off; keeping part of the
    base score makes the re-rank an adjustment rather than an override.
    """

    name = "lexical"

    def __init__(
        self,
        *,
        coverage_weight: float = 0.75,
        proximity_weight: float = 0.25,
        base_weight: float = 0.7,
    ) -> None:
        self.coverage_weight = coverage_weight
        self.proximity_weight = proximity_weight
        self.base_weight = min(max(base_weight, 0.0), 1.0)

    @staticmethod
    def _idf_weights(query_terms: Sequence[str], candidates: Sequence[Sequence[str]]) -> dict[str, float]:
        total = len(candidates) or 1
        df: Counter[str] = Counter()
        for terms in candidates:
            df.update(set(terms))
        weights: dict[str, float] = {}
        for term in set(query_terms):
            seen = df.get(term, 0)
            weights[term] = math.log(1.0 + total / (1.0 + seen))
        return weights

    @staticmethod
    def _normalise(base: Sequence[float]) -> list[float]:
        if not base:
            return []
        low, high = min(base), max(base)
        if high - low < 1e-12:
            return [1.0] * len(base)
        return [(value - low) / (high - low) for value in base]

    def score(
        self, query: str, candidates: Sequence[str], base: Sequence[float] | None = None
    ) -> list[float]:
        query_terms = tokenize(query)
        if not query_terms or not candidates:
            return [0.0] * len(candidates)

        tokenised = [tokenize(text) for text in candidates]
        weights = self._idf_weights(query_terms, tokenised)
        denominator = sum(weights.values()) or 1.0
        scores: list[float] = []

        for terms in tokenised:
            if not terms:
                scores.append(0.0)
                continue
            present = set(terms)
            covered = sum(weight for term, weight in weights.items() if term in present)
            coverage = covered / denominator

            # proximity: how many of the query terms appear anywhere at all,
            # normalised by chunk length so long chunks do not win by accident.
            hits = sum(1 for term in set(query_terms) if term in present)
            density = hits / math.sqrt(len(terms))
            scores.append(self.coverage_weight * coverage + self.proximity_weight * min(density, 1.0))

        if base is not None and len(base) == len(scores) and self.base_weight > 0:
            normalised = self._normalise([float(value) for value in base])
            keep = self.base_weight
            scores = [keep * norm + (1.0 - keep) * lex for norm, lex in zip(normalised, scores)]
        return scores


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder re-ranker (lazy import)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", *, device: str | None = None) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
            raise ModuleNotFoundError(
                "sentence-transformers is required for rerank=cross-encoder: "
                "pip install sentence-transformers (or use --rerank lexical)"
            ) from exc

        self.name = f"cross-encoder:{model_name}"
        self._model = CrossEncoder(model_name, device=device)

    def score(
        self, query: str, candidates: Sequence[str], base: Sequence[float] | None = None
    ) -> list[float]:
        if not candidates:
            return []
        pairs = [(query, text) for text in candidates]
        raw = self._model.predict(pairs)
        return [float(value) for value in raw]


def build_reranker(
    kind: str,
    model_name: str = "BAAI/bge-reranker-base",
    *,
    base_weight: float = 0.5,
) -> Reranker | None:
    kind = (kind or "none").lower()
    if kind in ("none", "off", ""):
        return None
    if kind == "lexical":
        return LexicalReranker(base_weight=base_weight)
    if kind in ("cross-encoder", "crossencoder", "cross_encoder"):
        return CrossEncoderReranker(model_name)
    raise ValueError(f"unknown reranker: {kind} (expected 'none', 'lexical' or 'cross-encoder')")
