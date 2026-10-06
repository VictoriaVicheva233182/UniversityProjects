"""Load the ledger, features and scores once and share them (CLI, copilot, API)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from ledgerlens.audit.features import build_features
from ledgerlens.audit.rules import RULES
from ledgerlens.config import AppConfig
from ledgerlens.data.generator import Ledger, load_ledger
from ledgerlens.ml.explain import Reason, explain


@dataclass
class Workspace:
    config: AppConfig
    ledger: Ledger
    features: pd.DataFrame
    scores: pd.DataFrame | None = None
    _index: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, config: AppConfig, with_scores: bool = True) -> Workspace:
        ledger = load_ledger(config.out / "data")
        features = build_features(ledger.entries, ledger.vendors, config.detection.approval_limit_eur)
        scores = None
        if with_scores:
            path = config.out / "scores.csv"
            if not path.is_file():
                raise FileNotFoundError(f"No scores in {path}. Run: ledgerlens analyze")
            scores = pd.read_csv(path, keep_default_na=False)
        ws = cls(config, ledger, features, scores)
        ws._index = {eid: i for i, eid in enumerate(ledger.entries["entry_id"])}
        return ws

    @property
    def out(self) -> Path:
        return self.config.out

    @cached_property
    def user_names(self) -> dict[str, str]:
        return dict(zip(self.ledger.users["user_id"], self.ledger.users["name"], strict=True))

    @cached_property
    def vendor_names(self) -> dict[str, str]:
        return dict(zip(self.ledger.vendors["vendor_id"], self.ledger.vendors["name"], strict=True))

    @cached_property
    def customer_names(self) -> dict[str, str]:
        return dict(zip(self.ledger.customers["customer_id"], self.ledger.customers["name"], strict=True))

    def position(self, entry_id: str) -> int:
        try:
            return self._index[entry_id]
        except KeyError as exc:
            raise KeyError(f"Unknown journal entry: {entry_id}") from exc

    def entry(self, entry_id: str) -> dict[str, Any]:
        return dict(self.ledger.entries.iloc[self.position(entry_id)])

    def feature_row(self, entry_id: str) -> dict[str, Any]:
        return dict(self.features.iloc[self.position(entry_id)])

    def reasons(self, entry_id: str, top: int = 4) -> list[Reason]:
        return explain(
            self.feature_row(entry_id),
            self.entry(entry_id),
            self.user_names,
            self.config.detection.approval_limit_eur,
            top,
        )

    def score_row(self, entry_id: str) -> dict[str, Any]:
        if self.scores is None:
            return {}
        return dict(self.scores.iloc[self.position(entry_id)])

    def rule_hits(self, entry_id: str) -> list[str]:
        hits = str(self.score_row(entry_id).get("rules", ""))
        return [RULES[h] for h in hits.split("|") if h in RULES]
