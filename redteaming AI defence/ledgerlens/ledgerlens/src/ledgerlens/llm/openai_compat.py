"""OpenAI-compatible chat completions client.

Works with the OpenAI API and with local servers that expose the same API
(LM Studio, vLLM, llama.cpp server). Set ``base_url`` for local servers.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import httpx

from ledgerlens.llm.base import LLMClient, LLMError, Message, with_retries


class OpenAICompatibleClient(LLMClient):
    name = "openai"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 900,
        timeout_s: float = 120.0,
        api_key: str | None = None,
    ) -> None:
        super().__init__(model, temperature, max_tokens)
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key and "api.openai.com" in self.base_url:
            raise LLMError("OPENAI_API_KEY is not set. Add it to your .env file.")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self._http = httpx.Client(timeout=timeout_s, headers=headers)

    def chat(self, messages: Sequence[Message], json_mode: bool = False) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        def _call() -> str:
            resp = self._http.post(f"{self.base_url}/chat/completions", json=payload)
            resp.raise_for_status()
            return str(resp.json()["choices"][0]["message"]["content"] or "")

        try:
            return with_retries(_call)
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenAI-compatible request failed: {exc}") from exc
