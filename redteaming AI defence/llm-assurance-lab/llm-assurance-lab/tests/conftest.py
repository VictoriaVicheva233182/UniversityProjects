"""Shared fixtures. All tests use the simulated provider, so no model or network is needed."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from assurance_lab.config import AppConfig, load_config
from assurance_lab.factory import build_pipeline
from assurance_lab.ml.dataset import generate_examples
from assurance_lab.ml.train import train_guardrail
from assurance_lab.rag.pipeline import AssistantPipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _project_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASSURANCE_HOME", str(ROOT))
    for key in (
        "ASSURANCE_LLM_PROVIDER",
        "ASSURANCE_LLM_MODEL",
        "ASSURANCE_LLM_BASE_URL",
        "ASSURANCE_JUDGE_ENABLED",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(scope="session")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("models") / "guardrail.joblib"
    train_guardrail(
        generate_examples(7, ROOT / "data/knowledge_base"), path, target_fpr=0.02, seed=7
    )
    return path


@pytest.fixture(scope="session")
def config_dir(tmp_path_factory: pytest.TempPathFactory, model_path: Path) -> Path:
    """Copies of the real configs that use the simulated LLM and the test model."""
    out = tmp_path_factory.mktemp("configs")
    for profile in ("baseline", "hardened"):
        data = yaml.safe_load((ROOT / f"configs/{profile}.yaml").read_text())
        data["llm"]["provider"] = "simulated"
        data["security"]["input_classifier"]["model_path"] = str(model_path)
        for key in ("knowledge_base_dir", "untrusted_sources_dir", "customers_file"):
            data["data"][key] = str(ROOT / data["data"][key])
        (out / f"{profile}.yaml").write_text(yaml.safe_dump(data))
    return out


@pytest.fixture()
def baseline_cfg(config_dir: Path) -> AppConfig:
    return load_config(config_dir / "baseline.yaml")


@pytest.fixture()
def hardened_cfg(config_dir: Path) -> AppConfig:
    return load_config(config_dir / "hardened.yaml")


@pytest.fixture()
def baseline(baseline_cfg: AppConfig) -> AssistantPipeline:
    return build_pipeline(baseline_cfg)


@pytest.fixture()
def hardened(hardened_cfg: AppConfig) -> AssistantPipeline:
    return build_pipeline(hardened_cfg)
