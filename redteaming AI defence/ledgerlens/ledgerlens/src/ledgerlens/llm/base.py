"""Provider-agnostic LLM interface."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

import httpx

logger = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant"]
T = TypeVar("T")


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMError(RuntimeError):
    """Raised when a provider call fails in a way the caller should see."""


class LLMClient(ABC):
    """Minimal chat interface. Every provider returns plain text."""

    name: str = "base"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 900) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def chat(self, messages: Sequence[Message], json_mode: bool = False) -> str:
        """Send a conversation and return the reply text. ``json_mode`` asks for JSON output."""

    def describe(self) -> str:
        return f"{self.name}:{self.model}"


def with_retries(fn: Callable[[], T], *, attempts: int = 3, base_delay: float = 1.0) -> T:
    """Retry transient HTTP failures (network errors, 429 and 5xx) with backoff."""
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status != 429 and status < 500:
                raise
            last_exc = exc
        except httpx.TransportError as exc:
            last_exc = exc
        if attempt < attempts:
            delay = base_delay * 2 ** (attempt - 1)
            logger.warning("LLM call failed (attempt %d/%d), retrying in %.1fs", attempt, attempts, delay)
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc
