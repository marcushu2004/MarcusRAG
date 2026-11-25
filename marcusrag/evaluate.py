"""Evaluation harness: turns "it feels better" into numbers.

Two layers, on purpose:

1. **Retrieval** metrics need no LLM at all — Recall@k, Hit@k and MRR are
   computed from the retrieved chunks against gold `(file, page)` labels.
2. **Answer** metrics need a generator — citation precision (are the cited
   pages among the gold pages), refusal accuracy (does the system refuse on
   unanswerable questions) and answer coverage.

Running the retrieval half is always possible (CI does exactly that with the
hash embedder); the answer half is skipped with an explicit `skipped` reason
when no LLM is reachable, rather than silently reporting 0.

Evalset format (JSONL):
    {"id": "q01", "question": "...", "gold": [{"file": "a.md", "page": 1}],
     "expect": "answer" | "refusal", "note": "why this case exists"}
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .citations import _normalize
from .config import Settings
from .pipeline import answer_question, build_index
from .retrieval import Retriever, build_retriever


@dataclass
class GoldRef:
    file: str
    page: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GoldRef":
        return cls(file=_normalize(str(payload["file"])), page=int(payload.get("page", 1)))

    def key(self) -> tuple[str, int]:
        return (self.file, self.page)


@dataclass
class EvalCase:
    id: str
    question: str
    gold: list[GoldRef] = field(default_factory=list)
    expect: str = "answer"
    note: str = ""


@dataclass
class CaseOutcome:
    id: str
    question: str
    expect: str
    retrieved: list[tuple[str, int]]
    first_hit_rank: int | None
    recall: float
    refused: bool | None = None
    citation_hit: float | None = None
    fabricated_citations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "expect": self.expect,
            "retrieved": [f"{file}:{page}" for file, page in self.retrieved],
            "first_hit_rank": self.first_hit_rank,
            "recall": round(self.recall, 4),
            "refused": self.refused,
            "citation_hit": None if self.citation_hit is None else round(self.citation_hit, 4),
            "fabricated_citations": self.fabricated_citations,
        }


@dataclass
class EvalReport:
    cases: list[CaseOutcome]
    k: int
    data_dir: str
    embedder: str
    search_type: str
    rerank: str
    skipped: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- aggregates
    @property
    def answerable(self) -> list["CaseOutcome"]:
        """Cases with a gold answer. Retrieval metrics ignore refusal cases."""
        return [c for c in self.cases if c.expect == "answer"]

    @property
    def recall_at_k(self) -> float:
        cases = self.answerable
        return sum(c.recall for c in cases) / len(cases) if cases else 0.0

    @property
    def hit_rate(self) -> float:
        cases = self.answerable
        hits = sum(1 for c in cases if c.first_hit_rank is not None)
        return hits / len(cases) if cases else 0.0

    @property
    def mrr(self) -> float:
        cases = self.answerable
        values = [1.0 / c.first_hit_rank if c.first_hit_rank else 0.0 for c in cases]
        return sum(values) / len(values) if values else 0.0

    @property
    def citation_precision(self) -> float | None:
        values = [c.citation_hit for c in self.cases if c.citation_hit is not None]
        return sum(values) / len(values) if values else None

    @property
    def refusal_accuracy(self) -> float | None:
        answerable = [c for c in self.cases if c.expect == "answer" and c.refused is not None]
        unanswerable = [c for c in self.cases if c.expect == "refusal" and c.refused is not None]
        if not answerable and not unanswerable:
            return None
        correct = sum(1 for c in answerable if not c.refused) + sum(1 for c in unanswerable if c.refused)
        return correct / (len(answerable) + len(unanswerable))

    @property
    def fabricated_citations(self) -> int:
        return sum(c.fabricated_citations for c in self.cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "cases": len(self.cases),
            "recall_at_k": round(self.recall_at_k, 4),
            "hit_rate": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "citation_precision": None if self.citation_precision is None else round(self.citation_precision, 4),
            "refusal_accuracy": None if self.refusal_accuracy is None else round(self.refusal_accuracy, 4),
            "fabricated_citations": self.fabricated_citations,
            "skipped": self.skipped,
            "config": {
                "data_dir": self.data_dir,
                "embedder": self.embedder,
                "search_type": self.search_type,
                "rerank": self.rerank,
            },
            "details": [case.to_dict() for case in self.cases],
            **self.extra,
        }

    def format(self) -> str:
        lines = [
            f"cases            : {len(self.cases)} ({len(self.answerable)} answerable, "
            f"{len(self.cases) - len(self.answerable)} unanswerable)",
            f"k                : {self.k}",
            f"config           : embedder={self.embedder} search={self.search_type} rerank={self.rerank}",
            f"Recall@{self.k}        : {self.recall_at_k:.3f}",
            f"Hit@{self.k}           : {self.hit_rate:.3f}",
            f"MRR              : {self.mrr:.3f}",
        ]
        if self.citation_precision is None:
            lines.append(f"citation precision: skipped ({self.skipped or 'no generator'})")
        else:
            lines.append(f"citation precision: {self.citation_precision:.3f}")
        if self.refusal_accuracy is None:
            lines.append(f"refusal accuracy : skipped ({self.skipped or 'no generator'})")
        else:
            lines.append(f"refusal accuracy : {self.refusal_accuracy:.3f}")
        lines.append(f"fabricated cites : {self.fabricated_citations}")
        misses = [c.id for c in self.answerable if c.first_hit_rank is None]
        if misses:
            lines.append(f"misses           : {', '.join(misses)}")
        return "\n".join(lines)


def load_evalset(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
        expect = payload.get("expect", "answer")
        if expect not in ("answer", "refusal"):
            raise ValueError(f"{path}:{line_number} has unknown expect={expect!r}")
        cases.append(
            EvalCase(
                id=str(payload.get("id") or f"c{line_number}"),
                question=str(payload["question"]),
                gold=[GoldRef.from_dict(item) for item in payload.get("gold", [])],
                expect=expect,
                note=str(payload.get("note", "")),
            )
        )
    if not cases:
        raise ValueError(f"{path} contains no evaluation cases")
    return cases


def _keys(chunks: Iterable) -> list[tuple[str, int]]:
    keys: list[tuple[str, int]] = []
    for chunk in chunks:
        file_name = _normalize(str(chunk.metadata.get("source", "")))
        try:
            page = int(chunk.metadata.get("page"))
        except (TypeError, ValueError):
            page = 1
        keys.append((file_name, page))
    return keys


def evaluate(
    settings: Settings,
    evalset_path: str | Path,
    *,
    retriever: Retriever | None = None,
    llm=None,
    k: int | None = None,
    rebuild: bool = True,
) -> EvalReport:
    """Run the evalset. `llm=None` evaluates retrieval only (no answer metrics).

    `rebuild=True` (default) drops and recreates the index so a run is
    reproducible and a corrupt index directory cannot poison the numbers.
    """
    cases = load_evalset(evalset_path)
    top_k = k or settings.top_k

    if retriever is None:
        from .embeddings import build_embedder
        from .vectorstore import build_store

        # Build the store exactly once and hand the *same* instance to both the
        # ingest and the query path. Opening a second PersistentClient on the
        # same directory inside one process makes chroma reuse its cached
        # system after the first one was reset, which surfaces as
        # "no such table: acquire_write".
        embedder = build_embedder(
            settings.embedder, settings.embedding_model, batch_size=settings.embedding_batch_size
        )
        if rebuild:
            shutil.rmtree(Path(settings.index_dir), ignore_errors=True)
        store = build_store(settings.store, settings.index_dir)
        build_index(settings, embedder=embedder, store=store, rebuild=True)
        retriever = build_retriever(store, embedder, settings)
    retriever.top_k = top_k

    outcomes: list[CaseOutcome] = []
    for case in cases:
        chunks = retriever.invoke(case.question)
        retrieved = _keys(chunks)
        gold_keys = {gold.key() for gold in case.gold}

        first_rank = None
        if gold_keys:
            for position, key in enumerate(retrieved, start=1):
                if key in gold_keys:
                    first_rank = position
                    break
            found = len(gold_keys & set(retrieved))
            recall = found / len(gold_keys)
        else:
            # unanswerable case: retrieval quality is judged by refusal instead
            recall = 1.0 if not retrieved else 0.0

        outcome = CaseOutcome(
            id=case.id,
            question=case.question,
            expect=case.expect,
            retrieved=retrieved,
            first_hit_rank=first_rank,
            recall=recall,
        )

        if llm is not None:
            result = answer_question(settings, case.question, retriever=retriever, llm=llm)
            outcome.refused = result.refused
            if result.citations and gold_keys:
                cited = {(c.file, c.page) for c in result.citations}
                outcome.citation_hit = len(cited & gold_keys) / len(cited)
            elif result.citations and not gold_keys:
                outcome.citation_hit = 0.0
            outcome.fabricated_citations = len(result.unverified_citations)

        outcomes.append(outcome)

    return EvalReport(
        cases=outcomes,
        k=top_k,
        data_dir=settings.data_dir,
        embedder=settings.embedder,
        search_type=settings.search_type,
        rerank=settings.rerank,
        skipped=None if llm is not None else "retrieval-only run (no generator)",
    )
