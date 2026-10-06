"""Write a short model card for the guardrail classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_model_card(metrics: dict[str, Any], path: Path) -> None:
    t, h = metrics["training"], metrics["held_out"]
    rows = "\n".join(
        f"| {name}{' (selected)' if name == t['model_name'] else ''} | {m['val_pr_auc']:.3f} | {m['val_roc_auc']:.3f} |"
        for name, m in t["comparison"].items()
    )
    groups = "\n".join(
        f"| {cat.replace('_', ' ')} | {m['n']} | {m['flag_rate'] * 100:.0f}% |"
        for cat, m in h["per_category"].items()
    )
    misses = (
        "\n".join(f"- `{m['text'][:110]}` (score {m['score']})" for m in h["misses"]) or "- none"
    )
    card = f"""# Model card: prompt injection guardrail

## Purpose
Flags user messages and untrusted document chunks that try to override the assistant's
instructions (prompt injection, jailbreaks, prompt extraction). It is one layer of defense,
not a complete solution. Data access risks are handled by access control, not by this model.

## Model
- Selected model: `{t["model_name"]}` (TF-IDF word and character n-grams)
- Decision threshold: {t["threshold"]:.3f}, chosen for false positive rate under {t["target_fpr"] * 100:.0f}% on validation data
- Trained: {t["trained_at_utc"]}, tool version {t["tool_version"]}

## Training data
{t["n_examples"]} synthetic examples, {t["n_malicious"]} malicious. Benign customer questions,
hard negatives that share attack vocabulary ("ignore", "instructions", "act as"), injection
templates and document snippets with hidden instructions. Split by template group.

## Model comparison (validation)
| Model | PR-AUC | ROC-AUC |
|---|---:|---:|
{rows}

Validation at threshold: precision {t["validation_metrics"]["precision"]:.3f}, recall
{t["validation_metrics"]["recall"]:.3f}, false positive rate {t["validation_metrics"]["false_positive_rate"]:.3f}.

## Held out evaluation (red team suite, written separately)
Recall {h["metrics"]["recall"]:.3f}, precision {h["metrics"]["precision"]:.3f}, false positive
rate {h["metrics"]["false_positive_rate"]:.3f} on {h["held_out_prompts"]} texts.

| Group | Texts | Flagged |
|---|---:|---:|
{groups}

Missed attacks (lowest scores first):
{misses}

## Limitations and safe use
- Trained on synthetic English and some Dutch text. Expect lower recall on other languages and
  on new attack styles. Retrain on reviewed production traffic before real use.
- An adaptive attacker can probe the threshold. Combine with access control and output checks.
- The model file is a pickle (joblib). Only load files you created yourself.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(card, encoding="utf-8")
