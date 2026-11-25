"""Citation parsing and verification.

`Evidence-Based` only means something if the citations in the answer are checked
against what was actually retrieved. This module turns "[paper.pdf:7]" style
references into structured data and flags fabricated ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CITATION_RE = re.compile(r"\[\s*([^\[\]\n]{1,160}?)\s*[:：]\s*(?:p\.?\s*)?(\d{1,4})\s*\]")


@dataclass(frozen=True)
class Citation:
    file: str
    page: int
    verified: bool

    def __str__(self) -> str:
        mark = "" if self.verified else " (unverified)"
        return f"[{self.file}:{self.page}]{mark}"


def _normalize(name: str) -> str:
    return Path(name.strip().strip("`'\"")).name.lower()


def parse_citations(text: str) -> list[Citation]:
    """Extract citations as (file, page) pairs; duplicates are collapsed."""
    seen: set[tuple[str, int]] = set()
    citations: list[Citation] = []
    for match in CITATION_RE.finditer(text or ""):
        file_name = _normalize(match.group(1))
        page = int(match.group(2))
        key = (file_name, page)
        if key in seen or not file_name:
            continue
        seen.add(key)
        citations.append(Citation(file=file_name, page=page, verified=False))
    return citations


def allowed_keys(chunks: Iterable) -> set[tuple[str, int]]:
    """The set of (file, page) pairs that exist in the retrieved evidence."""
    keys: set[tuple[str, int]] = set()
    for chunk in chunks:
        metadata = chunk.metadata
        file_name = _normalize(str(metadata.get("source", "")))
        try:
            page = int(metadata.get("page"))
        except (TypeError, ValueError):
            continue
        if file_name:
            keys.add((file_name, page))
    return keys


def verify_citations(citations: list[Citation], allowed: set[tuple[str, int]]) -> tuple[list[Citation], list[Citation]]:
    """Split citations into (verified, unverified)."""
    verified: list[Citation] = []
    unverified: list[Citation] = []
    for citation in citations:
        if (citation.file, citation.page) in allowed:
            verified.append(Citation(citation.file, citation.page, verified=True))
        else:
            unverified.append(citation)
    return verified, unverified


# --------------------------------------------------------------------------- #
# Evidence highlighting / back-linking
# --------------------------------------------------------------------------- #
# A verified citation is only useful if the reader can jump back to the evidence
# it points at. `build_evidence` numbers the retrieved chunks, marks which ones
# the answer actually cited, and extracts the sentences to show; `annotate_answer`
# rewrites the answer so every citation carries the matching evidence number.

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*|(?<=[.])\s+(?=[A-Z0-9\u4e00-\u9fff])")
_ANSI_HIGHLIGHT = "\x1b[7m"
_ANSI_BOLD = "\x1b[1m"
_ANSI_RESET = "\x1b[0m"


@dataclass(frozen=True)
class Evidence:
    """One retrieved chunk, with its citation state."""

    index: int
    file: str
    page: int | None
    score: float
    cited: bool
    snippet: str

    def label(self) -> str:
        return f"[{self.index}] {self.file}:{self.page}"

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "file": self.file,
            "page": self.page,
            "score": round(self.score, 4),
            "cited": self.cited,
            "snippet": self.snippet,
        }


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text or "") if part and part.strip()]


def _best_snippet(text: str, terms: set[str], *, limit: int = 260) -> str:
    """The sentence that shares most terms with the claim, else the opening one."""
    sentences = split_sentences(text)
    if not sentences:
        return " ".join((text or "").split())[:limit]
    if not terms:
        return sentences[0][:limit]
    scored = []
    for position, sentence in enumerate(sentences):
        lowered = sentence.lower()
        overlap = sum(1 for term in terms if term and term in lowered)
        scored.append((overlap, -position, sentence))
    scored.sort(reverse=True)
    return scored[0][2][:limit]


def build_evidence(
    chunks: Iterable,
    citations: list[Citation],
    *,
    query: str | None = None,
) -> list[Evidence]:
    """Number the evidence and mark which chunks the answer cited."""
    cited_keys = {(c.file, c.page) for c in citations if c.verified}
    terms = {token.lower() for token in re.split(r"[^\w\u4e00-\u9fff]+", query or "") if len(token) > 1}
    evidence: list[Evidence] = []
    for position, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata
        file_name = _normalize(str(metadata.get("source", "")))
        try:
            page = int(metadata.get("page"))
        except (TypeError, ValueError):
            page = None
        text = getattr(chunk, "text", "")
        evidence.append(
            Evidence(
                index=position,
                file=file_name,
                page=page,
                score=float(getattr(chunk, "score", 0.0)),
                cited=(file_name, page) in cited_keys,
                snippet=_best_snippet(text, terms),
            )
        )
    return evidence


def annotate_answer(answer: str, evidence: list[Evidence]) -> str:
    """Rewrite `[file:page]` into `[file:page | 证据#n]` so the claim links back."""
    lookup = {(item.file, item.page): item.index for item in evidence}
    if not lookup or not answer:
        return answer

    def _replace(match: re.Match[str]) -> str:
        file_name = _normalize(match.group(1))
        page = int(match.group(2))
        index = lookup.get((file_name, page))
        return f"[{file_name}:{page} | 证据#{index}]" if index else match.group(0)

    return CITATION_RE.sub(_replace, answer)


def highlight(text: str, query: str | None = None, *, color: bool = True) -> str:
    """Reverse-video the query terms inside `text` (terminal friendly)."""
    if not text:
        return ""
    terms = sorted(
        {t.lower() for t in re.split(r"[^\w\u4e00-\u9fff]+", query or "") if len(t) > 1},
        key=len,
        reverse=True,
    )
    if not color or not terms:
        return text
    pattern = re.compile("|".join(re.escape(term) for term in terms), re.IGNORECASE)
    return pattern.sub(lambda m: f"{_ANSI_HIGHLIGHT}{m.group(0)}{_ANSI_RESET}", text)


def format_evidence(evidence: list[Evidence], *, color: bool = False, query: str | None = None) -> str:
    """Human readable evidence block with cited/unused marks."""
    lines: list[str] = []
    for item in evidence:
        mark = "CITED " if item.cited else "unused"
        lines.append(f"{_ANSI_BOLD if color else ''}[{item.index}] {item.file}:{item.page}"
                     f"{_ANSI_RESET if color else ''} score={item.score:.3f} {mark}")
        lines.append("    " + highlight(item.snippet, query, color=color))
    return "\n".join(lines)

