"""MarcusRAG — evidence-based RAG over local documents.

Public API:
    from marcusrag import Settings, build_index, answer_question
"""

from __future__ import annotations

from .config import Settings, load_settings
from .pipeline import AnswerResult, IngestReport, answer_question, build_index, index_status

__all__ = [
    "Settings",
    "load_settings",
    "build_index",
    "answer_question",
    "index_status",
    "IngestReport",
    "AnswerResult",
]

__version__ = "0.2.0"
