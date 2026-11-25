"""Configuration objects for MarcusRAG.

Design rules:
- No module level side effects: importing this module must never touch the disk
  or build an index.
- Every tunable lives here and can be overridden by YAML or CLI flags.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


PROFILE_DEFAULTS: dict[str, dict[str, int]] = {
    "en": {"chunk_size": 600, "chunk_overlap": 100},
    "zh": {"chunk_size": 500, "chunk_overlap": 80},
}

# Known embedding models -> max sequence length (tokens).
# Used to warn when chunk_size will be silently truncated by the model.
MODEL_MAX_TOKENS: dict[str, int] = {
    "all-minilm-l6-v2": 256,
    "all-minilm-l12-v2": 256,
    "bge-small-en-v1.5": 512,
    "bge-base-en-v1.5": 512,
    "bge-small-zh-v1.5": 512,
    "bge-base-zh-v1.5": 512,
    "bge-m3": 8192,
    "multilingual-e5-small": 512,
    "multilingual-e5-base": 512,
    "text-embedding-3-small": 8192,
}

# Rough characters-per-token ratio used for the truncation warning.
CHARS_PER_TOKEN = {"en": 4.0, "zh": 1.4}


def model_slug(model_name: str) -> str:
    """Filesystem safe identifier for an embedding model."""
    cleaned = model_name.strip().replace("\\", "/").split("/")[-1]
    return "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in cleaned).strip("-") or "default"


def model_max_tokens(model_name: str) -> int | None:
    lowered = model_name.lower()
    for key, value in MODEL_MAX_TOKENS.items():
        if key in lowered:
            return value
    return None


@dataclass
class Settings:
    """Flat, serialisable configuration for the whole pipeline."""

    # data / index locations
    data_dir: str = "data/pdfs"
    index_root: str = "index"

    # embedding
    embedder: str = "hf"  # "hf" (sentence-transformers) | "hash" (offline fallback)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

    # chunking
    profile: str = "en"  # "en" | "zh" — controls default separators and ratios
    chunk_size: int = 600
    chunk_overlap: int = 100

    # retrieval
    store: str = "chroma"  # "chroma" (persistent) | "memory" (json, for tests/demo)
    search_type: str = "hybrid"  # "hybrid" (vector+BM25) | "mmr" | "similarity"
    top_k: int = 4
    fetch_k: int = 20
    mmr_lambda: float = 0.5
    hybrid_vector_weight: float = 1.0
    hybrid_bm25_weight: float = 1.0
    rrf_k: int = 60
    rerank: str = "lexical"  # "lexical" (no model) | "cross-encoder" | "none"
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_base_weight: float = 0.7  # lexical re-rank: share of the first-stage score kept

    # generation
    llm_backend: str = "ollama"  # "ollama" | "openai" | "fake"
    llm_model: str = "qwen2.5:3b"
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_api_key: str = ""  # falls back to $MARCUSRAG_API_KEY; never logged
    temperature: float = 0.0
    request_timeout: float = 120.0
    max_context_chars: int = 8000

    # evidence guard
    refuse_when_empty: bool = True
    verify_citations: bool = True

    # ---------------------------------------------------------------- helpers
    @property
    def index_dir(self) -> Path:
        """Per-embedding-model index directory.

        Keeping a separate directory per model avoids the dimension mismatch
        crash that happens when the same Chroma collection is reused with a
        different embedding model.
        """
        return Path(self.index_root) / model_slug(self.embedding_model) / self.store

    @property
    def manifest_path(self) -> Path:
        return Path(self.index_root) / model_slug(self.embedding_model) / "manifest.json"

    def warnings(self) -> list[str]:
        """Configuration sanity warnings surfaced by the CLI."""
        issues: list[str] = []
        if self.chunk_overlap >= self.chunk_size:
            issues.append(
                f"chunk_overlap ({self.chunk_overlap}) >= chunk_size ({self.chunk_size}); "
                "chunks will not advance and the index will explode."
            )
        max_tokens = model_max_tokens(self.embedding_model) if self.embedder == "hf" else 0
        if max_tokens:
            ratio = CHARS_PER_TOKEN.get(self.profile, 4.0)
            estimated = self.chunk_size / ratio
            if estimated > max_tokens * 0.9:
                issues.append(
                    f"chunk_size={self.chunk_size} (~{estimated:.0f} tokens at profile={self.profile}) "
                    f"exceeds the {max_tokens} token limit of {self.embedding_model}; "
                    f"the tail of every chunk will be silently truncated. "
                    f"Recommended: chunk_size<={int(max_tokens * ratio * 0.9)}."
                )
        if self.fetch_k < self.top_k:
            issues.append(f"fetch_k ({self.fetch_k}) < top_k ({self.top_k}); retrieval will be padded or short.")
        if self.search_type not in ("hybrid", "mmr", "similarity"):
            issues.append(f"unknown search_type '{self.search_type}'; expected hybrid | mmr | similarity.")
        if self.rerank not in ("none", "lexical", "cross-encoder"):
            issues.append(f"unknown rerank '{self.rerank}'; expected none | lexical | cross-encoder.")
        if not 0.0 <= self.rerank_base_weight <= 1.0:
            issues.append(
                f"rerank_base_weight={self.rerank_base_weight} is outside [0, 1]; "
                "the lexical re-ranker will clamp it."
            )
        if self.search_type == "hybrid" and self.hybrid_vector_weight + self.hybrid_bm25_weight <= 0:
            issues.append("hybrid_vector_weight + hybrid_bm25_weight must be > 0; falling back to equal weights.")
        if self.store == "chroma" and self.embedder == "hash":
            issues.append("store=chroma with embedder=hash is allowed for smoke tests but has poor recall quality.")
        backend = (self.llm_backend or "ollama").lower()
        if backend in ("openai", "openai-compatible", "compatible"):
            if not self.llm_base_url.rstrip("/").endswith("/v1") and ":11434" in self.llm_base_url:
                issues.append(
                    f"llm_backend=openai but llm_base_url={self.llm_base_url} lacks the '/v1' suffix; "
                    "Ollama's OpenAI shim lives at http://127.0.0.1:11434/v1."
                )
            if not self.llm_api_key and "127.0.0.1" not in self.llm_base_url and "localhost" not in self.llm_base_url:
                issues.append(
                    "llm_backend=openai with no api key for a remote endpoint; "
                    "set $MARCUSRAG_API_KEY or --llm-api-key, otherwise the request will likely return 401."
                )
        elif backend not in ("ollama", "fake"):
            issues.append(f"unknown llm_backend '{self.llm_backend}'; expected ollama | openai | fake.")
        return issues

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Settings":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")
        settings = cls(**payload)
        # If the caller only switched profile but kept default chunking, re-apply
        # profile defaults so zh/en documents get sensible chunk sizes.
        if settings.chunk_size == cls.chunk_size and settings.chunk_overlap == cls.chunk_overlap:
            defaults = PROFILE_DEFAULTS.get(settings.profile)
            if defaults:
                settings.chunk_size = defaults["chunk_size"]
                settings.chunk_overlap = defaults["chunk_overlap"]
        return settings

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
            raise ModuleNotFoundError("PyYAML is required to read config files: pip install pyyaml") from exc
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a YAML mapping at the top level")
        return cls.from_dict(payload)

    def override(self, **updates: Any) -> "Settings":
        """Return a copy with non-None overrides applied (used by the CLI)."""
        payload = self.to_dict()
        for key, value in updates.items():
            if value is not None and key in payload:
                payload[key] = value
        return Settings.from_dict(payload)


def load_settings(config_path: str | Path | None = None, **overrides: Any) -> Settings:
    """Load settings from `config.yaml` when present, otherwise use defaults.

    `config.example.yaml` is a template only and is never read implicitly —
    otherwise a fresh clone behaves differently from a fresh install.
    """
    if config_path:
        settings = Settings.from_yaml(config_path)
    elif Path("config.yaml").exists():
        settings = Settings.from_yaml("config.yaml")
    else:
        settings = Settings()
    settings = settings.override(**overrides)
    # Secrets must never be hard-coded in config.yaml; the environment wins when
    # the file leaves the key empty.
    if not settings.llm_api_key:
        settings.llm_api_key = os.environ.get("MARCUSRAG_API_KEY", "")
    return settings
