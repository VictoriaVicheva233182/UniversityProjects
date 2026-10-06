"""Build the assurance report (HTML and Markdown) from a baseline and a hardened run."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from assurance_lab.redteam.runner import load_run
from assurance_lab.reporting.frameworks import CONTROLS, LIMITATIONS, RECOMMENDATIONS, REGULATORY
from assurance_lab.reporting.risk import rate_risk

logger = logging.getLogger(__name__)

TEMPLATES = Path(__file__).parent / "templates"
RATING_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


def _truncate(text: str, n: int = 420) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _example(attacks: list[dict[str, Any]], category: str, success: bool) -> dict[str, Any] | None:
    matches = [a for a in attacks if a["category"] == category and a["success"] == success]
    matches.sort(key=lambda a: (a["converter"] != "none", a["run_id"]))
    if not matches:
        return None
    a = matches[0]
    return {
        "run_id": a["run_id"],
        "prompt": _truncate(a["prompt"], 300),
        "answer": _truncate(a["answer"]),
        "evidence": a.get("evidence"),
        "flags": a.get("flags", []),
    }


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def build_context(
    baseline_dir: Path, hardened_dir: Path, guardrail_metrics: dict[str, Any] | None
) -> dict[str, Any]:
    b_sum, b_att, _ = load_run(baseline_dir)
    h_sum, h_att, _ = load_run(hardened_dir)

    findings: list[dict[str, Any]] = []
    for key, b in b_sum["categories"].items():
        h = h_sum["categories"].get(key, {})
        before = rate_risk(b["impact"], b["asr"])
        after = rate_risk(b["impact"], h.get("asr", 0.0))
        findings.append(
            {
                "key": key,
                "id": f"F-{len(findings) + 1:02d}",
                "title": b["title"],
                "description": b["description"],
                "owasp": b["owasp"],
                "atlas": b["atlas"],
                "impact": b["impact"],
                "before": {
                    **b,
                    "risk": asdict(before),
                    "likelihood_label": before.likelihood_label,
                },
                "after": {**h, "risk": asdict(after), "likelihood_label": after.likelihood_label},
                "example_before": _example(b_att, key, True),
                "example_after": _example(h_att, key, True),
                "example_blocked": _example(h_att, key, False),
                "controls": CONTROLS.get(key, []),
                "recommendations": RECOMMENDATIONS.get(key, []),
            }
        )
    findings.sort(key=lambda f: (RATING_ORDER[f["before"]["risk"]["rating"]], -f["before"]["asr"]))
    for i, f in enumerate(findings, start=1):
        f["id"] = f"F-{i:02d}"

    high_before = sum(f["before"]["risk"]["rating"] in ("High", "Critical") for f in findings)
    high_after = sum(f["after"]["risk"]["rating"] in ("High", "Critical") for f in findings)
    residual = max(findings, key=lambda f: (f["after"]["risk"]["score"], f["after"]["asr"]))

    fbr_after = h_sum["benign"]["false_block_rate"]
    usability = (
        "so the defenses cost little usability"
        if fbr_after <= 0.05
        else "which is a real cost for customers and must be weighed against the risk reduction"
    )
    summary_points = [
        f"We ran {b_sum['attacks']['runs']} attack attempts in {len(findings)} risk categories against "
        f"the Havenkade customer assistant ({b_sum['target_llm']}).",
        f"Without controls, {_pct(b_sum['attacks']['asr'])} of attacks succeeded. With the hardened "
        f"configuration this fell to {_pct(h_sum['attacks']['asr'])}.",
        f"{high_before} of {len(findings)} risks were rated High or Critical before hardening, "
        f"{high_after} after.",
        f"The controls blocked or replaced {_pct(fbr_after)} of normal customer questions "
        f"(baseline {_pct(b_sum['benign']['false_block_rate'])}), {usability}.",
    ]
    if residual["after"]["asr"] > 0:
        summary_points.append(
            f"Highest remaining risk: {residual['title']} ({residual['after']['risk']['rating']}, "
            f"{_pct(residual['after']['asr'])} attack success)."
        )
    else:
        widest = max(findings, key=lambda f: f["after"]["ci95"][1])
        summary_points.append(
            "No attack succeeded against the hardened configuration in this test set. That does not "
            "prove the risks are gone: with this sample size the true success rate could still be up "
            f"to {_pct(widest['after']['ci95'][1])} ({widest['title']}, upper 95% bound)."
        )

    return {
        "date": date.today().isoformat(),
        "baseline": b_sum,
        "hardened": h_sum,
        "simulated": b_sum.get("simulated") or h_sum.get("simulated"),
        "findings": findings,
        "summary_points": summary_points,
        "guardrail": guardrail_metrics,
        "regulatory": REGULATORY,
        "limitations": LIMITATIONS,
        "pct": _pct,
        "baseline_dir": str(baseline_dir),
        "hardened_dir": str(hardened_dir),
    }


def write_report(
    baseline_dir: Path,
    hardened_dir: Path,
    out_dir: Path,
    guardrail_metrics_path: Path | None = None,
) -> tuple[Path, Path]:
    metrics = None
    if guardrail_metrics_path and guardrail_metrics_path.is_file():
        metrics = json.loads(guardrail_metrics_path.read_text(encoding="utf-8"))
    ctx = build_context(baseline_dir, hardened_dir, metrics)

    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "assurance_report.html"
    md_path = out_dir / "assurance_report.md"
    html_path.write_text(env.get_template("report.html.j2").render(**ctx), encoding="utf-8")
    md_env = Environment(
        loader=FileSystemLoader(TEMPLATES), autoescape=False, trim_blocks=True, lstrip_blocks=True
    )
    md_path.write_text(md_env.get_template("report.md.j2").render(**ctx), encoding="utf-8")
    logger.info("Report written to %s and %s", html_path, md_path)
    return html_path, md_path
