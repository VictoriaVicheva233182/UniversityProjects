"""Measure every method against the planted fraud. This is the only place labels are read."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ledgerlens.data.schemes import SCHEMES

METHODS = {
    "score_rules": "Classic rules, sorted by number of hits",
    "score_isolation_forest": "Isolation forest",
    "score_autoencoder": "Autoencoder",
    "score_ledgerlens": "LedgerLens score (autoencoder and rules combined)",
}


def _ranks(scores: pd.Series) -> np.ndarray:
    return np.asarray(scores.rank(ascending=False, method="first").astype(int))


def evaluate(scores: pd.DataFrame, truth: pd.DataFrame, top_k: list[int]) -> dict[str, Any]:
    scheme_of = dict(zip(truth["entry_id"], truth["scheme"], strict=True))
    y = scores["entry_id"].map(lambda i: i in scheme_of).to_numpy()
    schemes = scores["entry_id"].map(scheme_of).fillna("").to_numpy()
    n_fraud, n_schemes = int(y.sum()), len(SCHEMES)

    methods: dict[str, Any] = {}
    for col, label in METHODS.items():
        ranks = _ranks(scores[col])
        first_rank = {s: int(ranks[schemes == s].min()) for s in SCHEMES if (schemes == s).any()}
        at_k = []
        for k in top_k:
            top = ranks <= k
            at_k.append(
                {
                    "k": k,
                    "fraud_entries": int((top & y).sum()),
                    "precision": round(float((top & y).sum() / k), 4),
                    "recall": round(float((top & y).sum() / n_fraud), 4),
                    "schemes_found": int(sum(1 for r in first_rank.values() if r <= k)),
                }
            )
        methods[col] = {
            "label": label,
            "average_precision": round(float(average_precision_score(y, scores[col])), 4),
            "at_k": at_k,
            "first_rank_per_scheme": first_rank,
            "entries_to_find_all_schemes": max(first_rank.values()) if len(first_rank) == n_schemes else None,
        }

    flagged = scores["n_rules"].to_numpy() >= 1
    flagged2 = scores["n_rules"].to_numpy() >= 2
    rules_any = {
        "flagged_entries": int(flagged.sum()),
        "flagged_share": round(float(flagged.mean()), 4),
        "fraud_entries_flagged": int((flagged & y).sum()),
        "schemes_with_a_flag": len({s for s, f in zip(schemes, flagged, strict=True) if f and s}),
        "flagged_entries_2plus": int(flagged2.sum()),
        "schemes_with_2plus": len({s for s, f in zip(schemes, flagged2, strict=True) if f and s}),
    }
    return {
        "population": len(scores),
        "fraud_entries": n_fraud,
        "schemes": n_schemes,
        "rules_any_hit": rules_any,
        "methods": methods,
    }


def benford(amounts: pd.Series) -> dict[str, Any]:
    """First digit test on amounts of at least 10 euro, with Nigrini's MAD thresholds."""
    a = amounts[amounts >= 10].astype(float)
    first = a.astype(str).str.lstrip("0.").str[0].astype(int)
    observed = first.value_counts(normalize=True).reindex(range(1, 10), fill_value=0.0)
    expected = pd.Series({d: np.log10(1 + 1 / d) for d in range(1, 10)})
    mad = float((observed - expected).abs().mean())
    if mad <= 0.006:
        verdict = "close conformity"
    elif mad <= 0.012:
        verdict = "acceptable conformity"
    elif mad <= 0.015:
        verdict = "marginally acceptable conformity"
    else:
        verdict = "nonconformity"
    return {
        "n": len(a),
        "observed": [round(float(v), 4) for v in observed],
        "expected": [round(float(v), 4) for v in expected],
        "mad": round(mad, 4),
        "verdict": verdict,
    }
