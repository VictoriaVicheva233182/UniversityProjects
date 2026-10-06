"""Typed configuration loaded from YAML, with environment and CLI overrides.

Precedence (highest first): CLI arguments, environment variables (``LEDGERLENS_*``,
optionally from a ``.env`` file), the YAML file, defaults.
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
    model: str = "llama3.1:8b"
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 900
    timeout_s: float = 180.0


class DataConfig(BaseModel):
    seed: int = 11
    scale: float = Field(default=1.0, gt=0, le=3.0)  # 1.0 is about 200,000 journal entries
    output_dir: Path = Path("output")


class DetectionConfig(BaseModel):
    approval_limit_eur: float = 10_000.0
    top_k: list[int] = Field(default_factory=lambda: [25, 50, 100, 200, 500, 1000])
    isolation_forest_trees: int = 300
    autoencoder_epochs: int = 40


class CopilotConfig(BaseModel):
    top_n: int = 15
    max_steps: int = 6


class AppConfig(BaseModel):
    company: str = "Noordkade Logistics B.V."
    financial_year: int = 2025
    data: DataConfig = Field(default_factory=DataConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    copilot: CopilotConfig = Field(default_factory=CopilotConfig)

    @property
    def out(self) -> Path:
        return resolve_path(self.data.output_dir)


def project_root() -> Path:
    return Path(os.environ.get("LEDGERLENS_HOME", Path.cwd())).resolve()


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


def load_config(
    path: Path | str = "configs/default.yaml",
    *,
    provider: str | None = None,
    model: str | None = None,
) -> AppConfig:
    load_dotenv()
    cfg_path = resolve_path(path)
    data: dict[str, Any] = {}
    if cfg_path.is_file():
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    llm = data.setdefault("llm", {})
    for env_key, field in (
        ("LEDGERLENS_LLM_PROVIDER", "provider"),
        ("LEDGERLENS_LLM_MODEL", "model"),
        ("LEDGERLENS_LLM_BASE_URL", "base_url"),
    ):
        if os.environ.get(env_key):
            llm[field] = os.environ[env_key]
    if provider:
        llm["provider"] = provider
    if model:
        llm["model"] = model
    return AppConfig.model_validate(data)
