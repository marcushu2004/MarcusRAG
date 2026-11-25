"""Okapi BM25 over the exact same chunks that live in the vector store.

Why hand-rolled instead of `rank_bm25`:

- zero runtime dependency (BM25 is ~60 lines of arithmetic);
- CJK needs n-gram tokenisation, which the off-the-shelf implementations do not
  provide — an English word-splitter turns a Chinese sentence into one token and
  BM25 degenerates into exact-chunk matching;
- we need the *document ids* to line up with `Store` ids so the two rankings can
  be fused without a second lookup table.

The index is built from `Store.all_items()`, so it always describes the same
corpus the vectors describe — no index drift between the two halves of hybrid
retrieval.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_RUN = re.compile(r"[a-z0-9_]+")

STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "is", "are", "was", "were",
    "it", "this", "that", "for", "with", "as", "by", "at", "be", "from", "how", "what",
    "的", "了", "在", "是", "和", "与", "及", "对", "中", "上", "下", "吗", "呢", "把", "被",
}


def tokenize(text: str) -> list[str]:
    """Language-agnostic tokeniser: latin words + CJK character bigrams.

    Bigrams keep precision acceptable for Chinese without a dictionary
    (`文本切分` -> `文本`, `本切`, `切分`), and Latin words are left as words so
    identifiers such as `mmr_lambda` still match.
    """
    lowered = (text or "").lower()
    tokens: list[str] = []

    for run in _CJK_RUN.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))

    for run in _LATIN_RUN.findall(lowered):
        if run in STOPWORDS or len(run) < 2:
            continue
        tokens.append(run)
        # lightweight stemming: plurals and the most common suffix only
        if run.endswith("s") and len(run) > 3:
            tokens.append(run[:-1])

    return [t for t in tokens if t not in STOPWORDS]


class BM25Index:
    """In-memory Okapi BM25 (k1, b configurable)."""

    def __init__(self, items: Iterable[tuple[str, str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self._counts: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._df: Counter[str] = Counter()
        self._total_length = 0

        for doc_id, text in items:
            tokens = tokenize(text)
            counts = Counter(tokens)
            self.doc_ids.append(doc_id)
            self._counts.append(counts)
            self._lengths.append(len(tokens))
            self._total_length += len(tokens)
            self._df.update(counts.keys())

        self._size = len(self.doc_ids)
        self._avgdl = (self._total_length / self._size) if self._size else 0.0

    def __len__(self) -> int:
        return self._size

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (self._size - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        """Return `(doc_id, score)` for the `k` best chunks, best first."""
        if k <= 0 or not self._size:
            return []
        terms = tokenize(query)
        if not terms:
            return []

        scores: list[tuple[str, float]] = []
        avgdl = self._avgdl or 1.0
        for index, counts in enumerate(self._counts):
            score = 0.0
            length = self._lengths[index]
            norm = self.k1 * (1.0 - self.b + self.b * (length / avgdl))
            for term in terms:
                freq = counts.get(term)
                if not freq:
                    continue
                score += self._idf(term) * (freq * (self.k1 + 1.0)) / (freq + norm)
            if score > 0.0:
                scores.append((self.doc_ids[index], score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:k]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse several ranked id lists into one (RRF).

    RRF is used instead of score normalisation because cosine similarity and
    BM25 live on incomparable scales: min-max normalising them makes the result
    depend on the candidate set, while rank positions are already comparable.
    """
    weights = list(weights or [1.0] * len(rankings))
    if len(weights) != len(rankings):
        raise ValueError("weights must have one entry per ranking")

    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for position, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + position)

    return sorted(fused.items(), key=lambda item: item[1], reverse=True)
