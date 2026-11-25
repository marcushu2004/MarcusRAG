"""Backend tests for the chat models.

All HTTP is stubbed, so the suite stays offline: no Ollama daemon and no API key
are required to run it.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from marcusrag.config import Settings, load_settings
from marcusrag.llm import FakeChat, OllamaChat, OpenAIChat, build_llm


class _Response(io.BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


class StubbedHTTP(unittest.TestCase):
    """Base class that records outgoing requests and replays canned bodies."""

    def setUp(self) -> None:
        self.calls: list[dict] = []
        self.body: dict = {}
        self.status = 200
        self.raw = b"{}"
        patcher = mock.patch.object(urllib.request, "urlopen", self._urlopen)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _urlopen(self, request, timeout=None):
        self.calls.append(
            {
                "url": request.full_url,
                "headers": {k.lower(): v for k, v in request.header_items()},
                "payload": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        if self.status != 200:
            raise urllib.error.HTTPError(
                request.full_url, self.status, "error", {}, io.BytesIO(self.raw)
            )
        return _Response(json.dumps(self.body).encode("utf-8"))


class OllamaChatTests(StubbedHTTP):
    def test_posts_to_api_chat_and_strips_reply(self) -> None:
        self.body = {"message": {"content": "  带依据的回答 [1]  "}}

        reply = OllamaChat(model="qwen2.5:3b", base_url="http://127.0.0.1:11434/").generate("sys", "usr")

        self.assertEqual(reply, "带依据的回答 [1]")
        call = self.calls[0]
        self.assertEqual(call["url"], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(call["payload"]["model"], "qwen2.5:3b")
        self.assertFalse(call["payload"]["stream"])
        self.assertEqual(call["payload"]["messages"][0], {"role": "system", "content": "sys"})
        self.assertEqual(call["payload"]["messages"][1], {"role": "user", "content": "usr"})

    def test_unreachable_daemon_raises_actionable_error(self) -> None:
        with mock.patch.object(urllib.request, "urlopen", side_effect=urllib.error.URLError("refused")):
            with self.assertRaises(ConnectionError) as ctx:
                OllamaChat().generate("s", "u")
        self.assertIn("ollama serve", str(ctx.exception))


class OpenAIChatTests(StubbedHTTP):
    def test_builds_v1_url_with_bearer_token(self) -> None:
        self.body = {"choices": [{"message": {"role": "assistant", "content": "回答 [1]"}}]}

        chat = OpenAIChat(model="deepseek-chat", base_url="https://api.example.com/v1/", api_key="sk-test")
        self.assertEqual(chat.generate("sys", "usr"), "回答 [1]")

        call = self.calls[0]
        self.assertEqual(call["url"], "https://api.example.com/v1/chat/completions")
        self.assertEqual(call["headers"]["authorization"], "Bearer sk-test")
        self.assertEqual(call["payload"]["model"], "deepseek-chat")
        self.assertEqual(call["payload"]["messages"][0]["role"], "system")
        self.assertEqual(call["payload"]["temperature"], 0.0)

    def test_local_server_without_key_sends_no_auth(self) -> None:
        self.body = {"choices": [{"message": {"content": "ok"}}]}
        OpenAIChat(model="local", base_url="http://127.0.0.1:1234/v1").generate("s", "u")
        self.assertNotIn("authorization", self.calls[0]["headers"])

    def test_401_is_mapped_to_key_hint(self) -> None:
        self.status = 401
        self.raw = b'{"error":"invalid_api_key"}'
        with self.assertRaises(ConnectionError) as ctx:
            OpenAIChat(model="m", base_url="https://api.example.com/v1", api_key="bad").generate("s", "u")
        self.assertIn("401", str(ctx.exception))
        self.assertIn("API key", str(ctx.exception))

    def test_response_without_choices_is_an_error(self) -> None:
        self.body = {"error": "model_not_found"}
        with self.assertRaises(ConnectionError):
            OpenAIChat(model="m", base_url="http://127.0.0.1:1234/v1").generate("s", "u")


class BuildLLMTests(unittest.TestCase):
    def test_openai_aliases_resolve_to_openai_backend(self) -> None:
        for backend in ("openai", "openai-compatible", "compatible", "OPENAI"):
            with self.subTest(backend=backend):
                settings = Settings(
                    llm_backend=backend, llm_model="m", llm_base_url="http://x/v1", llm_api_key="k"
                )
                chat = build_llm(settings)
                self.assertIsInstance(chat, OpenAIChat)
                self.assertEqual((chat.model, chat.api_key), ("m", "k"))

    def test_ollama_and_fake_are_unchanged(self) -> None:
        self.assertIsInstance(build_llm(Settings()), OllamaChat)
        self.assertIsInstance(build_llm(Settings(llm_backend="fake")), FakeChat)

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            build_llm(Settings(llm_backend="not-a-backend"))
        self.assertIn("openai", str(ctx.exception))


class SettingsSecretTests(unittest.TestCase):
    def test_api_key_falls_back_to_environment(self) -> None:
        with mock.patch.dict("os.environ", {"MARCUSRAG_API_KEY": "sk-from-env"}):
            settings = load_settings(None, llm_backend="openai")
        self.assertEqual(settings.llm_api_key, "sk-from-env")

    def test_explicit_config_key_beats_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "c.yaml"
            config.write_text("llm_backend: openai\nllm_api_key: sk-from-file\n", encoding="utf-8")
            with mock.patch.dict("os.environ", {"MARCUSRAG_API_KEY": "sk-from-env"}):
                self.assertEqual(load_settings(str(config)).llm_api_key, "sk-from-file")

    def test_warnings_flag_openai_misconfiguration(self) -> None:
        no_v1 = Settings(llm_backend="openai", llm_base_url="http://127.0.0.1:11434")
        self.assertTrue(any("/v1" in w for w in no_v1.warnings()))

        no_key = Settings(llm_backend="openai", llm_base_url="https://api.example.com/v1", llm_api_key="")
        self.assertTrue(any("MARCUSRAG_API_KEY" in w for w in no_key.warnings()))

        good = Settings(llm_backend="openai", llm_base_url="http://127.0.0.1:1234/v1")
        self.assertFalse(any("llm_backend" in w for w in good.warnings()))

        bogus = Settings(llm_backend="bogus")
        self.assertTrue(any("unknown llm_backend" in w for w in bogus.warnings()))

    def test_key_is_not_part_of_warning_text(self) -> None:
        settings = Settings(llm_backend="openai", llm_base_url="https://api.example.com/v1", llm_api_key="sk-secret")
        self.assertNotIn("sk-secret", " ".join(settings.warnings()))


if __name__ == "__main__":
    unittest.main()
