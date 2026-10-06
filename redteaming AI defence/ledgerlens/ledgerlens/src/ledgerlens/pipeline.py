"""The steps of an engagement: generate, analyze, investigate, report."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from ledgerlens import __version__
from ledgerlens.audit.features import build_features
from ledgerlens.audit.rules import RULES, rule_summary, run_rules
from ledgerlens.config import AppConfig
from ledgerlens.copilot.agent import Investigation, investigate_top
from ledgerlens.data.generator import generate_ledger, load_ledger, save_ledger
from ledgerlens.llm.base import LLMClient
from ledgerlens.ml.detectors import score_all
from ledgerlens.ml.evaluate import benford, evaluate
from ledgerlens.workspace import Workspace

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def generate(cfg: AppConfig) -> dict[str, Any]:
    ledger = generate_ledger(cfg.data.seed, cfg.data.scale, cfg.detection.approval_limit_eur)
    save_ledger(ledger, cfg.out / "data")
    return {"entries": len(ledger.entries), "planted_fraud_entries": len(ledger.ground_truth)}


def analyze(cfg: AppConfig) -> dict[str, Any]:
    ledger = load_ledger(cfg.out / "data")
    features = build_features(ledger.entries, ledger.vendors, cfg.detection.approval_limit_eur)
    hits = run_rules(features)
    d = cfg.detection
    scores = score_all(features, hits, d.isolation_forest_trees, d.autoencoder_epochs, cfg.data.seed)
    scores.insert(2, "rules", rule_summary(hits).to_numpy())
    scores.to_csv(cfg.out / "scores.csv", index=False)

    metrics = evaluate(scores, ledger.ground_truth, d.top_k)
    purchases = ledger.entries.loc[ledger.entries["entry_type"] == "purchase_invoice", "amount"]
    metrics.update(
        {
            "company": cfg.company,
            "financial_year": cfg.financial_year,
            "seed": cfg.data.seed,
            "scale": cfg.data.scale,
            "tool_version": __version__,
            "analyzed_at_utc": _now(),
            "rule_hit_counts": {RULES[k]: int(v) for k, v in hits.sum().items()},
            "benford_purchase_invoices": benford(purchases),
            "users": len(ledger.users),
            "suppliers": len(ledger.vendors),
            "total_debits_eur": round(float(ledger.entries["amount"].sum()), 2),
        }
    )
    (cfg.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def copilot_metrics(results: list[Investigation], truth: dict[str, str], model: str, simulated: bool) -> dict[str, Any]:
    n = len(results)
    fraud = [r for r in results if r.entry_id in truth]
    legit = [r for r in results if r.entry_id not in truth]

    def share(items: list[Investigation], pred: Callable[[Investigation], bool]) -> float | None:
        return round(sum(pred(i) for i in items) / len(items), 4) if items else None

    return {
        "model": model,
        "simulated": simulated,
        "investigated": n,
        "grounded_share": share(results, lambda r: bool(r.grounding["grounded"])),
        "fallback_share": share(results, lambda r: r.fallback_used),
        "mean_tool_calls": round(sum(len(r.steps) for r in results) / n, 2) if n else 0,
        "mean_seconds": round(sum(r.seconds for r in results) / n, 1) if n else 0,
        "fraud_entries_investigated": len(fraud),
        "suspicious_on_fraud": share(fraud, lambda r: r.finding["assessment"] == "suspicious"),
        "scheme_correct_on_fraud": share(fraud, lambda r: r.finding["suspected_scheme"] == truth[r.entry_id]),
        "not_suspicious_on_legitimate": share(legit, lambda r: r.finding["assessment"] != "suspicious"),
        "investigated_at_utc": _now(),
    }


def investigate(cfg: AppConfig, llm: LLMClient, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    ws = Workspace.load(cfg)
    results = investigate_top(ws, llm, cfg.copilot.top_n, cfg.copilot.max_steps)
    with (cfg.out / "findings.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r.as_dict(), ensure_ascii=False, default=str) + "\n")
    truth = dict(zip(ws.ledger.ground_truth["entry_id"], ws.ledger.ground_truth["scheme"], strict=True))
    stats = copilot_metrics(results, truth, llm.describe(), cfg.llm.provider == "simulated")
    metrics_path = cfg.out / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    metrics["copilot"] = stats
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return stats


def load_findings(cfg: AppConfig) -> list[dict[str, Any]]:
    path = cfg.out / "findings.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
