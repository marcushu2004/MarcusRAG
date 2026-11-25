"""Chat model backends.

Deliberately dependency free at import time: both the Ollama and the
OpenAI-compatible backends talk HTTP with `urllib`, so a user with Ollama,
LM Studio, vLLM or any hosted chat API can use this project without installing
a provider SDK.
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


class OpenAIChat:
    """Minimal `/v1/chat/completions` client for any OpenAI-compatible server.

    Works with hosted APIs (base_url + api_key) and with local servers that
    ignore the key (LM Studio, vLLM, Ollama's `/v1` shim).
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:  # pragma: no cover - best effort diagnostics only
                detail = ""
            hint = " (missing or invalid API key?)" if exc.code in (401, 403) else ""
            raise ConnectionError(
                f"{self.base_url}/chat/completions returned HTTP {exc.code}{hint}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"cannot reach {self.base_url} ({exc}). "
                "Check --llm-base-url (for Ollama's OpenAI shim it is http://127.0.0.1:11434/v1)."
            ) from exc

        choices = body.get("choices") or []
        if not choices:
            raise ConnectionError(f"unexpected response without choices: {str(body)[:200]}")
        message = choices[0].get("message") or {}
        return str(message.get("content", "")).strip()


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


OPENAI_ALIASES = {"openai", "openai-compatible", "compatible"}


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
    if backend in OPENAI_ALIASES:
        return OpenAIChat(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            temperature=settings.temperature,
            timeout=settings.request_timeout,
        )
    raise ValueError(
        f"unknown llm backend: {backend} (expected 'ollama', 'openai' or 'fake')"
    )
