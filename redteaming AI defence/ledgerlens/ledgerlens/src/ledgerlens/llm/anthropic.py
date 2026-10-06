"""Anthropic Messages API client."""

from __future__ import annotations

import os
from collections.abc import Sequence

import httpx

from ledgerlens.llm.base import LLMClient, LLMError, Message, with_retries


class AnthropicClient(LLMClient):
    name = "anthropic"

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
        self.base_url = (base_url or "https://api.anthropic.com").rstrip("/")
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        self._http = httpx.Client(
            timeout=timeout_s,
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        )

    def chat(self, messages: Sequence[Message], json_mode: bool = False) -> str:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        payload: dict[str, object] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [m.as_dict() for m in messages if m.role != "system"],
        }
        if system:
            payload["system"] = system

        def _call() -> str:
            resp = self._http.post(f"{self.base_url}/v1/messages", json=payload)
            resp.raise_for_status()
            blocks = resp.json().get("content", [])
            return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")

        try:
            return with_retries(_call)
        except httpx.HTTPError as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc
