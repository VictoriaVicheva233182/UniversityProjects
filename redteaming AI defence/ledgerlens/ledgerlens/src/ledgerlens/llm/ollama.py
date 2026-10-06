"""Ollama client (local models, default for laptop runs)."""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from ledgerlens.llm.base import LLMClient, LLMError, Message, with_retries


class OllamaClient(LLMClient):
    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 900,
        timeout_s: float = 120.0,
    ) -> None:
        super().__init__(model, temperature, max_tokens)
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self._http = httpx.Client(timeout=timeout_s)

    def chat(self, messages: Sequence[Message], json_mode: bool = False) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        if json_mode:
            payload["format"] = "json"

        def _call() -> str:
            resp = self._http.post(f"{self.base_url}/api/chat", json=payload)
            if resp.status_code == 404:
                raise LLMError(f"Ollama does not know model '{self.model}'. Run: ollama pull {self.model}")
            resp.raise_for_status()
            return str(resp.json()["message"]["content"])

        try:
            return with_retries(_call)
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self.base_url}. Start it (https://ollama.com) "
                f"and pull the model with: ollama pull {self.model}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

    def list_models(self) -> list[str]:
        resp = self._http.get(f"{self.base_url}/api/tags")
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
