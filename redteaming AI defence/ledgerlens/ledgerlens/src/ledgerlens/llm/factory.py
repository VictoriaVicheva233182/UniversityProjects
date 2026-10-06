"""Create an LLM client from configuration."""

from __future__ import annotations

from ledgerlens.config import LLMConfig
from ledgerlens.llm.base import LLMClient, LLMError


def create_llm(cfg: LLMConfig) -> LLMClient:
    if cfg.provider == "simulated":
        from ledgerlens.llm.simulated import SimulatedLLM

        return SimulatedLLM()
    if cfg.provider == "ollama":
        from ledgerlens.llm.ollama import OllamaClient

        return OllamaClient(cfg.model, cfg.base_url, cfg.temperature, cfg.max_tokens, cfg.timeout_s)
    if cfg.provider == "openai":
        from ledgerlens.llm.openai_compat import OpenAICompatibleClient

        return OpenAICompatibleClient(cfg.model, cfg.base_url, cfg.temperature, cfg.max_tokens, cfg.timeout_s)
    if cfg.provider == "anthropic":
        from ledgerlens.llm.anthropic import AnthropicClient

        return AnthropicClient(cfg.model, cfg.base_url, cfg.temperature, cfg.max_tokens, cfg.timeout_s)
    raise LLMError(f"Unknown provider: {cfg.provider}")
