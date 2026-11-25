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
