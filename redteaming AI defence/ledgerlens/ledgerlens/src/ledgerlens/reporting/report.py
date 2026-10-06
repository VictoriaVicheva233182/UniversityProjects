"""Build the HTML report from the analysis metrics and the copilot findings."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ledgerlens.config import AppConfig
from ledgerlens.data.schemes import SCHEMES
from ledgerlens.ml.evaluate import METHODS
from ledgerlens.pipeline import load_findings

TEMPLATES = Path(__file__).parent / "templates"
SERIES_STYLE = {
    "score_ledgerlens": {"stroke": "#0A0A0A", "width": 3, "dash": ""},
    "score_rules": {"stroke": "#A21517", "width": 2, "dash": ""},
    "score_autoencoder": {"stroke": "#0A0A0A", "width": 1.5, "dash": "5 4"},
    "score_isolation_forest": {"stroke": "#9A9A9A", "width": 1.5, "dash": "2 4"},
}


def coverage_chart(metrics: dict[str, Any]) -> dict[str, Any]:
    """Line chart geometry: schemes found against list length, one line per method."""
    w, h, left, right, top, bottom = 640, 300, 44, 150, 16, 40
    ks = [a["k"] for a in metrics["methods"]["score_ledgerlens"]["at_k"]]
    n = metrics["schemes"]
    x = {k: left + i * (w - left - right) / (len(ks) - 1) for i, k in enumerate(ks)}

    def y(v: float) -> float:
        return top + (1 - v / n) * (h - top - bottom)

    series = []
    for key, m in metrics["methods"].items():
        pts = [(x[a["k"]], y(a["schemes_found"])) for a in m["at_k"]]
        series.append(
            {
                "key": key,
                "label": m["label"].split(" (")[0],
                "points": " ".join(f"{px:.1f},{py:.1f}" for px, py in pts),
                "end": pts[-1],
                **SERIES_STYLE.get(key, {"stroke": "#555", "width": 1, "dash": ""}),
            }
        )
    # Keep end labels from overlapping.
    series.sort(key=lambda s: s["end"][1])
    last = -99.0
    for s in series:
        s["label_y"] = max(s["end"][1], last + 15)
        last = s["label_y"]
    return {
        "w": w, "h": h, "series": series,
        "xticks": [{"x": x[k], "label": f"{k:,}"} for k in ks],
        "yticks": [{"y": y(v), "label": v} for v in range(0, n + 1, 2)],
        "left": left, "right": w - right, "top": top, "bottom": h - bottom,
    }  # fmt: skip


def benford_chart(b: dict[str, Any]) -> dict[str, Any]:
    w, h, left, bottom, top = 640, 220, 36, 34, 12
    peak = max(max(b["observed"]), max(b["expected"])) * 1.1
    bw = (w - left - 10) / 9
    bars = []
    for i, (o, e) in enumerate(zip(b["observed"], b["expected"], strict=True)):
        x = left + i * bw
        bars.append(
            {
                "digit": i + 1,
                "x": x + bw * 0.18,
                "w": bw * 0.64,
                "y": top + (1 - o / peak) * (h - top - bottom),
                "h": (o / peak) * (h - top - bottom),
                "ex": x + bw / 2,
                "ey": top + (1 - e / peak) * (h - top - bottom),
                "pct": f"{o * 100:.1f}%",
            }
        )
    return {
        "w": w,
        "h": h,
        "bars": bars,
        "base": h - bottom,
        "expected_points": " ".join(f"{b_['ex']:.1f},{b_['ey']:.1f}" for b_ in bars),
    }


def headline(metrics: dict[str, Any]) -> tuple[str, str]:
    ll = metrics["methods"]["score_ledgerlens"]
    rules = metrics["rules_any_hit"]
    n_s = metrics["schemes"]
    k_all = ll["entries_to_find_all_schemes"]
    if k_all is not None and k_all <= 1000:
        big = f"All {n_s} fraud schemes were in the first {k_all} entries."
    else:
        at100 = next(a for a in ll["at_k"] if a["k"] == 100)
        big = f"{at100['schemes_found']} of {n_s} fraud schemes were in the first 100 entries."
    small = (
        f"LedgerLens ranked {metrics['population']:,} journal entries by risk. The classic audit rules flagged "
        f"{rules['flagged_entries']:,} entries, which is the list an auditor would otherwise start from."
    )
    return big, small


def build_context(cfg: AppConfig) -> dict[str, Any]:
    metrics = json.loads((cfg.out / "metrics.json").read_text(encoding="utf-8"))
    findings = load_findings(cfg)
    big, small = headline(metrics)
    ll = metrics["methods"]["score_ledgerlens"]["first_rank_per_scheme"]
    rr = metrics["methods"]["score_rules"]["first_rank_per_scheme"]
    schemes = [{"key": k, **v, "rank_ledgerlens": ll.get(k), "rank_rules": rr.get(k)} for k, v in SCHEMES.items()]
    schemes.sort(key=lambda s: s["rank_ledgerlens"] or 10**9)
    methods = []
    for key in ("score_ledgerlens", "score_rules", "score_autoencoder", "score_isolation_forest"):
        m = metrics["methods"][key]
        at = {a["k"]: a for a in m["at_k"]}
        methods.append({"key": key, **m, "at": at})
    return {
        "m": metrics,
        "big": big,
        "small": small,
        "methods": methods,
        "method_labels": METHODS,
        "schemes": schemes,
        "coverage": coverage_chart(metrics),
        "benford": metrics["benford_purchase_invoices"],
        "benford_chart": benford_chart(metrics["benford_purchase_invoices"]),
        "findings": findings[:6],
        "copilot": metrics.get("copilot"),
        "date": date.today().isoformat(),
    }


def write_report(cfg: AppConfig) -> Path:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    path = cfg.out / "report.html"
    path.write_text(env.get_template("report.html.j2").render(**build_context(cfg)), encoding="utf-8")
    return path
