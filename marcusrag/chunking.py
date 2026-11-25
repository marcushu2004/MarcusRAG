"""Text chunking.

Fixes from the original codebase:
- dropped the bogus `from marshmallow.fields import List` import (an IDE
  mis-import that added an unrelated dependency);
- no module level `all_chunks` list that grows on every call;
- chunk size / overlap are validated instead of blindly trusted;
- separators are language aware (Chinese punctuation included).
"""

from __future__ import annotations

from typing import Iterable

from .documents import Doc

SEPARATORS = {
    "en": ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    "zh": ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
}


def split_documents(
    docs: Iterable[Doc],
    *,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    profile: str = "en",
) -> list[Doc]:
    """Split documents into overlapping chunks, preserving metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    separators = SEPARATORS.get(profile, SEPARATORS["en"])
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ModuleNotFoundError:  # lightweight fallback: fixed window with overlap
        return _fallback_split(docs, chunk_size, chunk_overlap)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        keep_separator=True,
    )

    chunks: list[Doc] = []
    for doc in docs:
        pieces = splitter.split_text(doc.text)
        for index, piece in enumerate(pieces):
            text = piece.strip()
            if not text:
                continue
            metadata = dict(doc.metadata)
            metadata["chunk_index"] = index
            chunks.append(Doc(text=text, metadata=metadata))
    return chunks


def _fallback_split(docs: Iterable[Doc], chunk_size: int, chunk_overlap: int) -> list[Doc]:
    chunks: list[Doc] = []
    step = chunk_size - chunk_overlap
    for doc in docs:
        text = doc.text
        for index, start in enumerate(range(0, max(len(text), 1), step)):
            piece = text[start : start + chunk_size].strip()
            if not piece:
                continue
            metadata = dict(doc.metadata)
            metadata["chunk_index"] = index
            chunks.append(Doc(text=piece, metadata=metadata))
            if start + chunk_size >= len(text):
                break
    return chunks
