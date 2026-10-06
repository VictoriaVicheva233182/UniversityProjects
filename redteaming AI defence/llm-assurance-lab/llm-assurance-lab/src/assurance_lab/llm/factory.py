"""Create an LLM client from configuration."""

from __future__ import annotations

from assurance_lab.config import JudgeConfig, LLMConfig
from assurance_lab.llm.base import LLMClient, LLMError


def create_llm(cfg: LLMConfig | JudgeConfig) -> LLMClient:
    temperature = getattr(cfg, "temperature", 0.0)
    max_tokens = getattr(cfg, "max_tokens", 400)
    provider = cfg.provider
    if provider == "simulated":
        from assurance_lab.llm.simulated import SimulatedLLM

        return SimulatedLLM()
    if provider == "ollama":
        from assurance_lab.llm.ollama import OllamaClient

        return OllamaClient(cfg.model, cfg.base_url, temperature, max_tokens, cfg.timeout_s)
    if provider == "openai":
        from assurance_lab.llm.openai_compat import OpenAICompatibleClient

        return OpenAICompatibleClient(
            cfg.model, cfg.base_url, temperature, max_tokens, cfg.timeout_s
        )
    if provider == "anthropic":
        from assurance_lab.llm.anthropic import AnthropicClient

        return AnthropicClient(cfg.model, cfg.base_url, temperature, max_tokens, cfg.timeout_s)
    raise LLMError(f"Unknown provider: {provider}")
