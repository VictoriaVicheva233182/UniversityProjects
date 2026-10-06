"""Runtime wrapper around the trained prompt injection classifier.

Security note: the model is stored with joblib (pickle). Only load model files
you produced yourself. Never load a model file from an untrusted source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class GuardrailUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class GuardrailDecision:
    flagged: bool
    score: float
    threshold: float


class InputClassifier:
    def __init__(self, bundle: dict[str, Any], threshold: float | None = None) -> None:
        self.pipeline = bundle["pipeline"]
        self.threshold = float(threshold if threshold is not None else bundle["threshold"])
        self.model_name = str(bundle.get("model_name", "unknown"))
        self.metadata = {k: v for k, v in bundle.items() if k != "pipeline"}

    @classmethod
    def load(cls, path: Path, threshold: float | None = None) -> InputClassifier:
        if not path.is_file():
            raise GuardrailUnavailable(
                f"Guardrail model not found at {path}. Train it first with: assurance-lab train"
            )
        return cls(joblib.load(path), threshold)

    def raw_scores(self, texts: list[str]) -> np.ndarray:
        """Model probability for each full text."""
        return np.asarray(self.pipeline.predict_proba(texts)[:, 1])

    def scores(self, texts: list[str]) -> np.ndarray:
        """Attack score per text: the maximum over the full text and each of its sentences.

        Scoring sentences (and adjacent sentence pairs) separately defeats dilution, where an attacker pads an attack
        with harmless text (an urgent story, a fake authority claim) to pull the
        average score of the whole message under the threshold.
        """
        segments: list[str] = []
        owners: list[int] = []
        for i, text in enumerate(texts):
            sentences = _sentences(text)
            pairs = [f"{a} {b}" for a, b in pairwise(sentences)]
            parts = list(dict.fromkeys([text, *sentences, *pairs]))
            segments += parts
            owners += [i] * len(parts)
        seg_scores = self.raw_scores(segments)
        out = np.zeros(len(texts))
        for owner, score in zip(owners, seg_scores, strict=True):
            out[owner] = max(out[owner], score)
        return out

    def check(self, text: str) -> GuardrailDecision:
        score = float(self.scores([text])[0])
        return GuardrailDecision(score >= self.threshold, score, self.threshold)


def _sentences(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if len(p.strip()) >= 12]
