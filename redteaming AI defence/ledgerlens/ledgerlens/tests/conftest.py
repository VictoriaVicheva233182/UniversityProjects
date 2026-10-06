"""A small ledger (about 20,000 entries) analysed once per test session, fully offline."""

from __future__ import annotations

from pathlib import Path

import pytest

from ledgerlens.config import AppConfig
from ledgerlens.llm.simulated import SimulatedLLM
from ledgerlens.pipeline import analyze, generate, investigate
from ledgerlens.workspace import Workspace


@pytest.fixture(scope="session")
def cfg(tmp_path_factory: pytest.TempPathFactory) -> AppConfig:
    c = AppConfig()
    c.data.output_dir = Path(tmp_path_factory.mktemp("out"))
    c.data.scale = 0.1
    c.detection.isolation_forest_trees = 60
    c.detection.autoencoder_epochs = 15
    c.llm.provider = "simulated"
    c.copilot.top_n = 4
    c.out.mkdir(parents=True, exist_ok=True)
    generate(c)
    analyze(c)
    investigate(c, SimulatedLLM())
    return c


@pytest.fixture(scope="session")
def ws(cfg: AppConfig) -> Workspace:
    return Workspace.load(cfg)
