"""Ingest and query orchestration.

Everything here is a pure function of explicit arguments: no module level state,
no import time disk writes. Each collaborator (embedder, store, retriever, llm)
can be injected, which is what makes the pipeline testable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .chunking import split_documents
from .citations import Citation, allowed_keys, parse_citations, verify_citations
from .config import Settings
from .documents import Doc, discover_files, file_digest, load_documents
from .embeddings import Embedder, build_embedder
from .llm import ChatModel, build_llm
from .prompts import build_messages
from .retrieval import Retrieved, build_retriever
from .vectorstore import Store, build_store

NO_EVIDENCE = "文档中未找到相关依据。"


@dataclass
class IngestReport:
    files: int
    documents: int
    chunks: int
    written: int
    skipped_files: int
    index_dir: str
    store: str
    warnings: list[str] = field(default_factory=list)
    per_source: dict[str, int] = field(default_factory=dict)


@dataclass
class AnswerResult:
    question: str
    answer: str
    sources: list[Retrieved]
    citations: list[Citation]
    unverified_citations: list[Citation]
    refused: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "refused": self.refused,
            "reason": self.reason,
            "sources": [
                {
                    "file": Path(s.source).name,
                    "page": s.page,
                    "score": round(s.score, 4),
                    "preview": " ".join(s.text.split())[:160],
                }
                for s in self.sources
            ],
            "citations": [str(c) for c in self.citations],
            "unverified_citations": [str(c) for c in self.unverified_citations],
        }


def chunk_id(chunk: Doc) -> str:
    """Deterministic id: re-indexing the same text never creates duplicates."""
    source = Path(str(chunk.metadata.get("source", ""))).name
    page = chunk.metadata.get("page", 0)
    index = chunk.metadata.get("chunk_index", 0)
    fingerprint = hashlib.sha1(f"{source}|{page}|{index}|{chunk.text[:120]}".encode("utf-8")).hexdigest()
    return f"c_{fingerprint[:24]}"


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_index(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    store: Store | None = None,
    docs: list[Doc] | None = None,
    rebuild: bool = False,
    incremental: bool = True,
    verbose: bool = False,
) -> IngestReport:
    files = discover_files(settings.data_dir)
    if docs is None:
        docs = load_documents(settings.data_dir, verbose=verbose)

    embedder = embedder or build_embedder(
        settings.embedder, settings.embedding_model, batch_size=settings.embedding_batch_size
    )
    store = store or build_store(settings.store, settings.index_dir)

    if rebuild:
        store.reset()

    manifest = {} if rebuild else _read_manifest(settings.manifest_path)
    known_hashes: dict[str, str] = manifest.get("files", {})

    changed_docs = docs
    skipped = 0
    if incremental and not rebuild and known_hashes:
        filtered: list[Doc] = []
        for doc in docs:
            source = str(doc.metadata.get("source", ""))
            path = Path(source)
            if not path.exists():
                filtered.append(doc)
                continue
            if known_hashes.get(str(path)) == file_digest(path):
                skipped += 1
                continue
            filtered.append(doc)
        if filtered:
            changed_docs = filtered
        else:
            changed_docs = []

    chunks = split_documents(
        changed_docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        profile=settings.profile,
    )

    written = 0
    if chunks:
        vectors = embedder.embed_documents([c.text for c in chunks])
        written = store.upsert(
            ids=[chunk_id(c) for c in chunks],
            texts=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
            embeddings=vectors,
        )

    hashes = {}
    for path in files:
        try:
            hashes[str(path)] = file_digest(path)
        except OSError:
            continue
    _write_manifest(
        settings.manifest_path,
        {
            "embedding_model": settings.embedding_model,
            "store": settings.store,
            "files": hashes,
            "chunks": store.count(),
        },
    )

    per_source: dict[str, int] = {}
    for chunk in chunks:
        name = Path(str(chunk.metadata.get("source", "unknown"))).name
        per_source[name] = per_source.get(name, 0) + 1

    return IngestReport(
        files=len(files),
        documents=len(docs),
        chunks=len(chunks),
        written=written,
        skipped_files=skipped,
        index_dir=str(settings.index_dir),
        store=settings.store,
        warnings=settings.warnings(),
        per_source=per_source,
    )


def answer_question(
    settings: Settings,
    question: str,
    *,
    retriever=None,
    llm: ChatModel | None = None,
    embedder: Embedder | None = None,
    store: Store | None = None,
) -> AnswerResult:
    if retriever is None:
        embedder = embedder or build_embedder(
            settings.embedder, settings.embedding_model, batch_size=settings.embedding_batch_size
        )
        store = store or build_store(settings.store, settings.index_dir)
        retriever = build_retriever(store, embedder, settings)

    llm = llm or build_llm(settings)
    sources: list[Retrieved] = list(retriever.invoke(question))

    if not sources:
        if settings.refuse_when_empty:
            return AnswerResult(
                question=question,
                answer=NO_EVIDENCE,
                sources=[],
                citations=[],
                unverified_citations=[],
                refused=True,
                reason="索引中没有可用证据（可能尚未建立索引，或索引为空）",
            )
        return AnswerResult(question, NO_EVIDENCE, [], [], [], True, "no candidates")

    system, user = build_messages(question, sources, max_chars=settings.max_context_chars)
    answer = llm.generate(system, user).strip() or NO_EVIDENCE

    citations = parse_citations(answer)
    unverified: list[Citation] = []
    if settings.verify_citations and citations:
        verified, unverified = verify_citations(citations, allowed_keys(sources))
        citations = verified

    # A refusal is detected on the opening clause only, so a model that prefixes
    # its output ("(fake) 文档中未找到…") is still classified correctly.
    head = answer[:80]
    refused = (NO_EVIDENCE in head) or ("未找到相关依据" in head)
    return AnswerResult(
        question=question,
        answer=answer,
        sources=sources,
        citations=citations,
        unverified_citations=unverified,
        refused=refused,
        reason=None,
    )


def index_status(settings: Settings) -> dict[str, Any]:
    manifest = _read_manifest(settings.manifest_path)
    return {
        "index_dir": str(settings.index_dir),
        "store": settings.store,
        "embedding_model": settings.embedding_model,
        "manifest_files": len(manifest.get("files", {})),
        "manifest_chunks": manifest.get("chunks", 0),
        "data_dir": str(settings.data_dir),
        "data_files": [p.name for p in discover_files(settings.data_dir)],
        "warnings": settings.warnings(),
    }
