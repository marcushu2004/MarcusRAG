"""Vector store backends.

Both stores share one tiny interface so the pipeline can be unit tested without
Chroma installed. Chroma is imported lazily inside `ChromaStore.__init__`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Protocol, Sequence


class Store(Protocol):
    def upsert(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
        embeddings: Sequence[Sequence[float]],
    ) -> int: ...

    def query(self, embedding: Sequence[float], k: int) -> list[tuple[str, str, dict[str, Any], float]]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma only accepts scalar metadata values."""
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif value is None:
            continue
        else:
            clean[key] = str(value)
    return clean


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class ChromaStore:
    """Persistent Chroma collection (cosine space)."""

    collection_name = "marcusrag"

    def __init__(self, directory: str | Path) -> None:
        try:
            import chromadb  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
            raise ModuleNotFoundError(
                "chromadb is required for store=chroma: pip install chromadb "
                "(or use --store memory for a dependency free demo)"
            ) from exc

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.directory))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, ids, texts, metadatas, embeddings) -> int:
        if not ids:
            return 0
        payload = {
            "ids": list(ids),
            "documents": list(texts),
            "metadatas": [_clean_metadata(m) for m in metadatas],
            "embeddings": [list(map(float, e)) for e in embeddings],
        }
        self._collection.upsert(**payload)
        return len(payload["ids"])

    def query(self, embedding, k: int) -> list[tuple[str, str, dict[str, Any], float]]:
        if k <= 0 or self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(map(float, embedding))],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits: list[tuple[str, str, dict[str, Any], float]] = []
        for doc_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            # cosine distance -> similarity
            hits.append((doc_id, text or "", metadata or {}, 1.0 - float(distance)))
        hits.sort(key=lambda item: item[3], reverse=True)
        return hits

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


class MemoryStore:
    """JSON backed in-process store; zero heavy dependencies."""

    filename = "memory_store.json"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path = self.directory / self.filename
        self._items: dict[str, dict[str, Any]] = {}
        if self._path.exists():
            try:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
                self._items = {item["id"]: item for item in payload.get("items", [])}
            except (json.JSONDecodeError, KeyError):
                self._items = {}

    def _flush(self) -> None:
        payload = {"items": list(self._items.values())}
        self._path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def upsert(self, ids, texts, metadatas, embeddings) -> int:
        written = 0
        for doc_id, text, metadata, embedding in zip(ids, texts, metadatas, embeddings):
            self._items[doc_id] = {
                "id": doc_id,
                "text": text,
                "metadata": _clean_metadata(metadata),
                "embedding": list(map(float, embedding)),
            }
            written += 1
        if written:
            self._flush()
        return written

    def query(self, embedding, k: int) -> list[tuple[str, str, dict[str, Any], float]]:
        scored = [
            (item["id"], item["text"], item["metadata"], _cosine(embedding, item["embedding"]))
            for item in self._items.values()
        ]
        scored.sort(key=lambda item: item[3], reverse=True)
        return scored[: max(k, 0)]

    def count(self) -> int:
        return len(self._items)

    def reset(self) -> None:
        self._items = {}
        self._flush()


def build_store(kind: str, directory: str | Path) -> Store:
    kind = (kind or "chroma").lower()
    if kind == "chroma":
        return ChromaStore(directory)
    if kind == "memory":
        return MemoryStore(directory)
    raise ValueError(f"unknown store: {kind} (expected 'chroma' or 'memory')")
