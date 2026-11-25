"""Document loading with explicit, testable behaviour.

Fixes from the original codebase:
- no module level `all_docs` list that silently accumulates across imports;
- every loaded file is accumulated (the old loop overwrote `docs` and only the
  last PDF reached the splitter);
- supported extensions are declared, not implicitly assumed;
- missing/empty inputs raise or return empty lists instead of indexing [0].
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


@dataclass
class Doc:
    """Minimal document unit: text plus provenance metadata."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))

    @property
    def page(self) -> int | None:
        value = self.metadata.get("page")
        return int(value) if isinstance(value, (int, float, str)) and str(value).isdigit() else None


def file_digest(path: Path) -> str:
    sha = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _load_pdf(path: Path) -> tuple[list[Doc], int]:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise ModuleNotFoundError("pypdf is required for PDF input: pip install pypdf") from exc

    reader = PdfReader(str(path))
    docs: list[Doc] = []
    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:  # a single broken page must not kill the whole ingest
            text = ""
        text = text.strip()
        if not text:
            continue
        docs.append(Doc(text=text, metadata={"source": str(path), "page": index + 1, "type": "pdf"}))
    return docs, len(reader.pages)


def _load_text(path: Path) -> list[Doc]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    return [Doc(text=text, metadata={"source": str(path), "page": 1, "type": path.suffix.lstrip(".")})]


def discover_files(data_dir: str | Path) -> list[Path]:
    """Sorted list of supported files; empty when the directory is missing."""
    directory = Path(data_dir)
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)


def load_documents(data_dir: str | Path, *, verbose: bool = False) -> list[Doc]:
    """Load every supported file under `data_dir`.

    The returned list contains documents from *all* files. This is the bug the
    original `loaders.py` had: `docs` was reassigned inside the loop, so only
    the final PDF was chunked and indexed.

    Sources that produced no text are reported by `scan_documents`; use it when
    the caller needs to tell the user "this PDF was skipped" instead of silently
    indexing one document fewer.
    """
    return scan_documents(data_dir, verbose=verbose).docs


@dataclass
class LoadResult:
    """Documents plus the sources that yielded no usable text."""

    docs: list[Doc] = field(default_factory=list)
    empty_sources: list[tuple[str, str]] = field(default_factory=list)


def scan_documents(data_dir: str | Path, *, verbose: bool = False) -> LoadResult:
    """Like `load_documents`, but also records why a file contributed nothing.

    A scanned PDF has no text layer: pypdf returns empty strings for every page.
    Dropping it silently makes the index quietly incomplete, so the reason is
    carried back to the caller instead.
    """
    result = LoadResult()
    for path in discover_files(data_dir):
        is_pdf = path.suffix.lower() == ".pdf"
        if is_pdf:
            loaded, total_pages = _load_pdf(path)
        else:
            loaded, total_pages = _load_text(path), 1
        if verbose:
            print(f"  loaded {path.name}: {len(loaded)} page(s) / segment(s)")
        if not loaded:
            if is_pdf and total_pages:
                reason = f"{total_pages} 页均无可提取文本层（疑似扫描件/图片型 PDF，需要 OCR）"
            elif is_pdf:
                reason = "PDF 无任何页面"
            else:
                reason = "文件为空或仅含空白字符"
            result.empty_sources.append((path.name, reason))
        result.docs.extend(loaded)  # accumulate, never overwrite
    return result


def group_sources(docs: Iterable[Doc]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for doc in docs:
        name = Path(doc.source).name
        counts[name] = counts.get(name, 0) + 1
    return counts
