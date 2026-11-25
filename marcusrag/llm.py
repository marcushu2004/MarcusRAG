"""Chat model backends.

Deliberately dependency free at import time: the Ollama backend talks to the
local HTTP API with `urllib`, so a user with Ollama running can use this project
without installing any LangChain provider package.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol


class ChatModel(Protocol):
    def generate(self, system: str, user: str) -> str: ...


class OllamaChat:
    """Minimal Ollama /api/chat client."""

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        *,
        base_url: str = "http://127.0.0.1:11434",
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"cannot reach Ollama at {self.base_url} ({exc}). "
                f"Start it with `ollama serve` and pull the model with `ollama pull {self.model}`."
            ) from exc
        return (body.get("message") or {}).get("content", "").strip()


class FakeChat:
    """Deterministic model used by tests and by `--llm-backend fake` demos."""

    def __init__(self, reply: str = "（fake 模型）文档中未找到相关依据。") -> None:
        self.reply = reply
        self.last_system: str | None = None
        self.last_user: str | None = None

    def generate(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return self.reply


def build_llm(settings) -> ChatModel:
    backend = (settings.llm_backend or "ollama").lower()
    if backend == "fake":
        return FakeChat()
    if backend == "ollama":
        return OllamaChat(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            temperature=settings.temperature,
            timeout=settings.request_timeout,
        )
    raise ValueError(f"unknown llm backend: {backend} (expected 'ollama' or 'fake')")
