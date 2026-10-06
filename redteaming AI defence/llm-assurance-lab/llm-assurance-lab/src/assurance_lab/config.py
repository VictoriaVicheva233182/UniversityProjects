"""Typed configuration loaded from YAML, with environment and CLI overrides.

Precedence (highest first): explicit CLI arguments, environment variables
(``ASSURANCE_*``, optionally from a ``.env`` file), the YAML profile, defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

ProviderName = Literal["ollama", "openai", "anthropic", "simulated"]


class LLMConfig(BaseModel):
    provider: ProviderName = "ollama"
    model: str = "llama3.2:3b"
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 400
    timeout_s: float = 120.0


class DataConfig(BaseModel):
    knowledge_base_dir: Path = Path("data/knowledge_base")
    untrusted_sources_dir: Path | None = Path("data/untrusted_sources")
    customers_file: Path = Path("data/customers.json")


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=3, ge=1, le=20)
    chunk_max_chars: int = Field(default=700, ge=100)
    enforce_access_control: bool = False


class InputClassifierConfig(BaseModel):
    enabled: bool = False
    model_path: Path = Path("models/guardrail.joblib")
    screen_retrieved_chunks: bool = False
    threshold: float | None = None  # None means: use the threshold stored with the model


class OutputFilterConfig(BaseModel):
    enabled: bool = False
    allowed_domains: list[str] = Field(default_factory=lambda: ["havenkade.example"])
    grounding_check: bool = True


class SecurityConfig(BaseModel):
    hardened_prompt: bool = False
    authorize_customer_lookup: bool = False
    sanitize_untrusted_context: bool = False
    input_classifier: InputClassifierConfig = Field(default_factory=InputClassifierConfig)
    output_filter: OutputFilterConfig = Field(default_factory=OutputFilterConfig)


class JudgeConfig(BaseModel):
    enabled: bool = False
    provider: ProviderName = "ollama"
    model: str = "llama3.2:3b"
    base_url: str | None = None
    timeout_s: float = 120.0


class AppConfig(BaseModel):
    profile: str = "baseline"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)


def project_root() -> Path:
    """Directory that relative paths are resolved against (``ASSURANCE_HOME`` or cwd)."""
    return Path(os.environ.get("ASSURANCE_HOME", Path.cwd())).resolve()


def resolve_path(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def load_dotenv(path: Path | None = None) -> None:
    """Minimal ``.env`` loader. Existing environment variables always win."""
    env_file = path or project_root() / ".env"
    if not env_file.is_file():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if value and key not in os.environ:
            os.environ[key] = value


def _env_overrides(data: dict[str, Any]) -> None:
    mapping = {
        "ASSURANCE_LLM_PROVIDER": ("llm", "provider"),
        "ASSURANCE_LLM_MODEL": ("llm", "model"),
        "ASSURANCE_LLM_BASE_URL": ("llm", "base_url"),
        "ASSURANCE_JUDGE_ENABLED": ("judge", "enabled"),
        "ASSURANCE_JUDGE_PROVIDER": ("judge", "provider"),
        "ASSURANCE_JUDGE_MODEL": ("judge", "model"),
        "ASSURANCE_JUDGE_BASE_URL": ("judge", "base_url"),
    }
    for env_key, (section, field) in mapping.items():
        value = os.environ.get(env_key)
        if value:
            data.setdefault(section, {})[field] = value


def load_config(
    path: Path | str,
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> AppConfig:
    """Load a YAML profile and apply environment and CLI overrides."""
    load_dotenv()
    cfg_path = resolve_path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    data: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    _env_overrides(data)
    llm = data.setdefault("llm", {})
    if provider:
        llm["provider"] = provider
        # A simulated target makes no sense with a real judge unless asked for.
        if provider == "simulated":
            data.setdefault("judge", {})["enabled"] = False
    if model:
        llm["model"] = model
    if base_url:
        llm["base_url"] = base_url
    return AppConfig.model_validate(data)
