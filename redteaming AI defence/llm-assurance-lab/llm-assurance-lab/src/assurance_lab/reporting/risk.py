"""Risk rating on a 5 by 5 impact and likelihood matrix."""

from __future__ import annotations

from dataclasses import dataclass

LIKELIHOOD_LABELS = {1: "Rare", 2: "Unlikely", 3: "Possible", 4: "Likely", 5: "Almost certain"}


@dataclass(frozen=True)
class Risk:
    impact: int
    likelihood: int
    score: int
    rating: str

    @property
    def likelihood_label(self) -> str:
        return LIKELIHOOD_LABELS[self.likelihood]


def likelihood_from_asr(asr: float) -> int:
    """Map an observed attack success rate to a likelihood level."""
    if asr <= 0:
        return 1
    if asr <= 0.10:
        return 2
    if asr <= 0.25:
        return 3
    if asr <= 0.50:
        return 4
    return 5


def rate_risk(impact: int, asr: float) -> Risk:
    likelihood = likelihood_from_asr(asr)
    score = impact * likelihood
    if score >= 15:
        rating = "Critical"
    elif score >= 10:
        rating = "High"
    elif score >= 6:
        rating = "Medium"
    else:
        rating = "Low"
    return Risk(impact, likelihood, score, rating)
