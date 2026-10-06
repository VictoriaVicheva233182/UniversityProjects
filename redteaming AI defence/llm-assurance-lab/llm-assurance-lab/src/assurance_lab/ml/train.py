"""Train, compare and calibrate the guardrail classifier.

Steps:
1. Split the data by template group (no near duplicates across the split).
2. Compare a few interpretable models on the validation split (PR-AUC).
3. Pick the decision threshold with the highest recall whose false positive
   rate stays under a business target (default 2%), because wrongly blocking
   real customers has a cost too.
4. Refit the chosen model on all data and save it with its threshold and metrics.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from assurance_lab import __version__
from assurance_lab.ml.dataset import Example

logger = logging.getLogger(__name__)


def _features() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, lowercase=True),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2
                ),
            ),
        ]
    )


def candidate_models(seed: int) -> dict[str, Pipeline]:
    return {
        "tfidf_logreg": Pipeline(
            [
                ("features", _features()),
                (
                    "clf",
                    LogisticRegression(
                        C=4.0, class_weight="balanced", max_iter=3000, random_state=seed
                    ),
                ),
            ]
        ),
        "tfidf_linear_svm": Pipeline(
            [
                ("features", _features()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(C=0.5, class_weight="balanced", random_state=seed), cv=3
                    ),
                ),
            ]
        ),
        "tfidf_complement_nb": Pipeline(
            [("features", _features()), ("clf", ComplementNB(alpha=0.3))]
        ),
    }


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """Lowest threshold whose false positive rate on validation stays <= target_fpr.

    Recall can only go down as the threshold goes up, so the lowest threshold that
    meets the false positive budget is also the one with the highest recall. Taking
    the lowest one (instead of any threshold with the same validation recall) leaves
    more margin for attacks that look different from the training data.
    """
    neg = np.sort(scores[y_true == 0])[::-1]
    if len(neg) == 0:
        return 0.5
    allowed = int(np.floor(target_fpr * len(neg)))
    threshold = float(neg[allowed]) + 1e-6 if allowed < len(neg) else float(neg[-1])
    return float(min(max(threshold, 0.05), 0.95))


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_guardrail(
    examples: list[Example],
    out_path: Path,
    *,
    target_fpr: float = 0.02,
    seed: int = 42,
) -> dict[str, Any]:
    texts = np.array([e.text for e in examples], dtype=object)
    labels = np.array([e.label for e in examples])
    groups = np.array([e.group for e in examples])

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, val_idx = next(splitter.split(texts, labels, groups))
    logger.info("Training examples: %d train, %d validation", len(train_idx), len(val_idx))

    comparison: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    for name, model in candidate_models(seed).items():
        model.fit(list(texts[train_idx]), labels[train_idx])
        scores = model.predict_proba(list(texts[val_idx]))[:, 1]
        comparison[name] = {
            "val_pr_auc": round(float(average_precision_score(labels[val_idx], scores)), 4),
            "val_roc_auc": round(float(roc_auc_score(labels[val_idx], scores)), 4),
        }
        fitted[name] = model
        logger.info(
            "%-22s PR-AUC %.3f  ROC-AUC %.3f",
            name,
            comparison[name]["val_pr_auc"],
            comparison[name]["val_roc_auc"],
        )

    best_name = max(
        comparison, key=lambda n: (comparison[n]["val_pr_auc"], comparison[n]["val_roc_auc"])
    )
    val_scores = fitted[best_name].predict_proba(list(texts[val_idx]))[:, 1]
    threshold = choose_threshold(labels[val_idx], val_scores, target_fpr)
    val_metrics = binary_metrics(labels[val_idx], val_scores, threshold)
    logger.info(
        "Selected %s with threshold %.3f (val recall %.3f, FPR %.3f)",
        best_name,
        threshold,
        val_metrics["recall"],
        val_metrics["false_positive_rate"],
    )

    final = candidate_models(seed)[best_name]
    final.fit(list(texts), labels)

    bundle = {
        "pipeline": final,
        "model_name": best_name,
        "threshold": threshold,
        "target_fpr": target_fpr,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool_version": __version__,
        "n_examples": len(examples),
        "n_malicious": int(labels.sum()),
        "comparison": comparison,
        "validation_metrics": val_metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    logger.info("Saved guardrail model to %s", out_path)
    return {k: v for k, v in bundle.items() if k != "pipeline"}
