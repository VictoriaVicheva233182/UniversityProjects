"""Unsupervised anomaly detectors. No labels are used anywhere in training.

- Isolation forest: entries that are easy to isolate with random splits are unusual.
- Autoencoder: a small neural network learns to rebuild normal entries; entries it
  rebuilds badly are unusual.
- LedgerLens score: the average rank of the autoencoder and the rule count, so audit
  knowledge (rules) and learned unusualness (model) both count. The isolation forest
  is kept as a comparison; it struggles with these mostly yes or no features.

The combination was chosen on a development ledger (seed 7) and checked on a fresh
test ledger (seed 11, the default), so it is not tuned to the data it is reported on.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from ledgerlens.audit.features import MODEL_FEATURES

logger = logging.getLogger(__name__)


def _matrix(f: pd.DataFrame) -> np.ndarray:
    return f[MODEL_FEATURES].to_numpy(dtype=float)


def isolation_forest_scores(f: pd.DataFrame, trees: int = 300, seed: int = 7) -> np.ndarray:
    model = IsolationForest(n_estimators=trees, max_samples=1024, random_state=seed, n_jobs=-1)
    x = _matrix(f)
    model.fit(x)
    return -model.score_samples(x)


def autoencoder_scores(f: pd.DataFrame, epochs: int = 40, seed: int = 7) -> np.ndarray:
    x = StandardScaler().fit_transform(_matrix(f))
    x = np.clip(x, -8, 8)
    model = MLPRegressor(
        hidden_layer_sizes=(16, 6, 16), activation="relu", batch_size=1024,
        max_iter=epochs, learning_rate_init=0.003, random_state=seed, tol=1e-5,
    )  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x, x)
    return np.asarray(((model.predict(x) - x) ** 2).mean(axis=1))


def rank_pct(scores: np.ndarray) -> np.ndarray:
    return np.asarray(pd.Series(scores).rank(pct=True, method="average"))


def score_all(f: pd.DataFrame, rule_hits: pd.DataFrame, trees: int, epochs: int, seed: int) -> pd.DataFrame:
    logger.info("Training isolation forest on %d entries", len(f))
    iso = isolation_forest_scores(f, trees, seed)
    logger.info("Training autoencoder")
    ae = autoencoder_scores(f, epochs, seed)
    n_rules = rule_hits.sum(axis=1).to_numpy()
    # Tie break the rule count by amount, the way auditors usually sort a rule hit list.
    rules = n_rules + rank_pct(f["log_amount"].to_numpy()) * 0.5
    scores = pd.DataFrame(
        {
            "entry_id": f["entry_id"],
            "n_rules": n_rules,
            "score_rules": rules,
            "score_isolation_forest": iso,
            "score_autoencoder": ae,
        }
    )
    scores["score_ledgerlens"] = (rank_pct(ae) + rank_pct(rules)) / 2
    scores["rank"] = scores["score_ledgerlens"].rank(ascending=False, method="first").astype(int)
    return scores
