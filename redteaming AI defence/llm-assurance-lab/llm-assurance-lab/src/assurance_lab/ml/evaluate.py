"""Held-out evaluation of the guardrail on the red team suite.

The classifier is meant to stop instruction attacks (prompt injection,
jailbreaks, prompt extraction). Requests for other customers' data or internal
documents are excluded on purpose: those must be stopped by access control,
because their wording is often indistinguishable from a normal question.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from assurance_lab.guardrails.input_classifier import InputClassifier
from assurance_lab.ml.train import binary_metrics
from assurance_lab.rag.documents import Chunk
from assurance_lab.redteam.attacks import AttackSuite, BenignSet
from assurance_lab.redteam.converters import DEFAULT_CONVERTERS
from assurance_lab.redteam.runner import expand_cases

INSTRUCTION_ATTACKS = {
    "system_prompt_leakage",
    "direct_prompt_injection",
    "jailbreak_policy_violation",
}
BENIGN_LOOKING = {"indirect_prompt_injection", "misinformation"}


def evaluate_on_suite(
    classifier: InputClassifier,
    suite: AttackSuite,
    benign: BenignSet,
    chunks: list[Chunk],
) -> dict[str, Any]:
    texts: list[str] = []
    labels: list[int] = []
    cats: list[str] = []

    def add(text: str, label: int, category: str) -> None:
        texts.append(text)
        labels.append(label)
        cats.append(category)

    for _run_id, _conv, prompt, case in expand_cases(suite, DEFAULT_CONVERTERS):
        if case.category in INSTRUCTION_ATTACKS:
            add(prompt, 1, case.category)
        elif case.category in BENIGN_LOOKING:
            add(prompt, 0, "benign_looking_attack_prompt")
    for q in benign.questions:
        add(q.prompt, 0, "benign_customer_question")

    y = np.array(labels)
    scores = classifier.scores(texts)
    metrics = binary_metrics(y, scores, classifier.threshold)

    per_category: dict[str, dict[str, Any]] = {}
    for cat in sorted(set(cats)):
        idx = [i for i, c in enumerate(cats) if c == cat]
        flagged = int((scores[idx] >= classifier.threshold).sum())
        per_category[cat] = {
            "n": len(idx),
            "flagged": flagged,
            "flag_rate": round(flagged / len(idx), 4),
        }

    misses = [
        {"text": texts[i], "score": round(float(scores[i]), 3)}
        for i in np.argsort(scores)
        if y[i] == 1 and scores[i] < classifier.threshold
    ][:10]
    false_alarms = [
        {"text": texts[i], "score": round(float(scores[i]), 3)}
        for i in np.argsort(-scores)
        if y[i] == 0 and scores[i] >= classifier.threshold
    ][:10]

    untrusted = [c for c in chunks if c.trust == "untrusted"]
    trusted = [c for c in chunks if c.trust == "trusted"]
    chunk_scores_u = classifier.scores([c.text for c in untrusted]) if untrusted else np.array([])
    chunk_scores_t = classifier.scores([c.text for c in trusted]) if trusted else np.array([])

    return {
        "threshold": classifier.threshold,
        "model_name": classifier.model_name,
        "held_out_prompts": len(texts),
        "metrics": metrics,
        "per_category": per_category,
        "misses": misses,
        "false_alarms": false_alarms,
        "chunk_screening": {
            "untrusted_chunks": [
                {
                    "id": c.id,
                    "score": round(float(s), 3),
                    "flagged": bool(s >= classifier.threshold),
                }
                for c, s in zip(untrusted, chunk_scores_u, strict=True)
            ],
            "trusted_chunks_flagged": int((chunk_scores_t >= classifier.threshold).sum()),
            "trusted_chunks_total": len(trusted),
        },
    }
