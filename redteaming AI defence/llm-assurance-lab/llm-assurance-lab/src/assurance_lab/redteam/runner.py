"""Run the attack suite and the benign set against one pipeline profile."""

from __future__ import annotations

import json
import logging
import platform
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from assurance_lab import __version__
from assurance_lab.llm.base import LLMError
from assurance_lab.rag.pipeline import AssistantPipeline, AssistantResponse
from assurance_lab.redteam.attacks import AttackSuite, BenignSet
from assurance_lab.redteam.converters import get_converter
from assurance_lab.redteam.detectors import DetectorRegistry
from assurance_lab.redteam.judge import LLMJudge
from assurance_lab.redteam.scoring import rate, wilson_interval

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]


@dataclass
class AttackResult:
    run_id: str
    case_id: str
    category: str
    converter: str
    prompt: str
    answer: str
    success: bool
    detection_method: str
    evidence: str | None
    blocked: bool
    block_reason: str | None
    flags: list[str]
    sources: list[dict[str, Any]]
    latency_ms: float
    error: str | None = None


@dataclass
class BenignResult:
    id: str
    prompt: str
    answer: str
    blocked: bool
    block_reason: str | None
    keyword_hit: bool
    latency_ms: float
    error: str | None = None


@dataclass
class RunOutput:
    attacks: list[AttackResult] = field(default_factory=list)
    benign: list[BenignResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def expand_cases(suite: AttackSuite, converters: list[str]) -> list[tuple[str, str, str, Any]]:
    """Return (run_id, converter, prompt, case) for every case and applicable converter."""
    runs = []
    for case in suite.cases:
        runs.append((f"{case.id}", "none", case.prompt, case))
        if case.apply_converters:
            for name in converters:
                runs.append((f"{case.id}:{name}", name, get_converter(name)(case.prompt), case))
    return runs


def _safe_answer(
    pipeline: AssistantPipeline, prompt: str, customer_id: str
) -> tuple[AssistantResponse, str | None]:
    try:
        return pipeline.answer(prompt, customer_id), None
    except LLMError:
        raise  # a dead model invalidates the whole run, so stop loudly
    except Exception as exc:
        logger.exception("Case failed: %s", prompt[:60])
        return AssistantResponse(answer=""), f"{type(exc).__name__}: {exc}"


def run_profile(
    pipeline: AssistantPipeline,
    suite: AttackSuite,
    benign: BenignSet | None,
    *,
    converters: list[str],
    judge: LLMJudge | None = None,
    limit: int | None = None,
    categories: list[str] | None = None,
    progress: ProgressFn | None = None,
) -> RunOutput:
    detectors = DetectorRegistry(pipeline.customers, suite.session_customer_id, judge)
    runs = expand_cases(suite, converters)
    if categories:
        runs = [r for r in runs if r[3].category in categories]
    if limit:
        runs = runs[:limit]

    out = RunOutput()
    for run_id, converter, prompt, case in runs:
        if progress:
            progress(run_id)
        resp, error = _safe_answer(pipeline, prompt, suite.session_customer_id)
        det = detectors.detect(case.detector, prompt, resp)
        out.attacks.append(
            AttackResult(
                run_id=run_id,
                case_id=case.id,
                category=case.category,
                converter=converter,
                prompt=prompt,
                answer=resp.answer,
                success=det.success and error is None,
                detection_method=det.method,
                evidence=det.evidence,
                blocked=resp.blocked,
                block_reason=resp.block_reason,
                flags=resp.flags,
                sources=[asdict(s) for s in resp.sources],
                latency_ms=resp.latency_ms,
                error=error,
            )
        )

    if benign is not None:
        questions = benign.questions[:limit] if limit else benign.questions
        for q in questions:
            if progress:
                progress(q.id)
            resp, error = _safe_answer(pipeline, q.prompt, benign.session_customer_id)
            answer = resp.answer.lower()
            out.benign.append(
                BenignResult(
                    id=q.id,
                    prompt=q.prompt,
                    answer=resp.answer,
                    blocked=resp.blocked,
                    block_reason=resp.block_reason,
                    keyword_hit=any(e in answer for e in q.expect_any) if q.expect_any else True,
                    latency_ms=resp.latency_ms,
                    error=error,
                )
            )

    out.summary = summarize(out, suite, pipeline, converters, judge)
    return out


def summarize(
    out: RunOutput,
    suite: AttackSuite,
    pipeline: AssistantPipeline,
    converters: list[str],
    judge: LLMJudge | None,
) -> dict[str, Any]:
    by_cat: dict[str, list[AttackResult]] = defaultdict(list)
    for r in out.attacks:
        by_cat[r.category].append(r)

    categories = {}
    for key, cat in suite.categories.items():
        results = by_cat.get(key, [])
        n, k = len(results), sum(r.success for r in results)
        low, high = wilson_interval(k, n)
        categories[key] = {
            "title": cat.title,
            "owasp": cat.owasp,
            "atlas": cat.atlas,
            "impact": cat.impact,
            "description": cat.description,
            "runs": n,
            "successes": k,
            "asr": round(rate(k, n), 4),
            "ci95": [round(low, 4), round(high, 4)],
            "blocked": sum(r.blocked for r in results),
        }

    total, succ = len(out.attacks), sum(r.success for r in out.attacks)
    nb = len(out.benign)
    benign_blocked = sum(b.blocked for b in out.benign)
    cfg = pipeline.config
    return {
        "profile": cfg.profile,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool_version": __version__,
        "python": platform.python_version(),
        "target_llm": pipeline.llm.describe(),
        "simulated": cfg.llm.provider == "simulated",
        "judge": judge.llm.describe() if judge else None,
        "converters": converters,
        "security_controls": cfg.security.model_dump(mode="json"),
        "access_control": cfg.retrieval.enforce_access_control,
        "attacks": {
            "runs": total,
            "successes": succ,
            "asr": round(rate(succ, total), 4),
            "ci95": [round(x, 4) for x in wilson_interval(succ, total)],
            "errors": sum(r.error is not None for r in out.attacks),
        },
        "categories": categories,
        "benign": {
            "questions": nb,
            "blocked": benign_blocked,
            "false_block_rate": round(rate(benign_blocked, nb), 4),
            "keyword_hit_rate": round(rate(sum(b.keyword_hit for b in out.benign), nb), 4),
            "mean_latency_ms": round(sum(b.latency_ms for b in out.benign) / nb, 1) if nb else 0.0,
        },
    }


def save_run(out: RunOutput, runs_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = runs_dir / f"{out.summary['profile']}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "attacks.jsonl").open("w", encoding="utf-8") as fh:
        for r in out.attacks:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    with (run_dir / "benign.jsonl").open("w", encoding="utf-8") as fh:
        for b in out.benign:
            fh.write(json.dumps(asdict(b), ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(out.summary, indent=2), encoding="utf-8")
    # Plain text pointer instead of a symlink, so it also works on Windows.
    (runs_dir / f"latest_{out.summary['profile']}.txt").write_text(str(run_dir), encoding="utf-8")
    return run_dir


def latest_run(runs_dir: Path, profile: str) -> Path | None:
    pointer = runs_dir / f"latest_{profile}.txt"
    if pointer.is_file():
        path = Path(pointer.read_text(encoding="utf-8").strip())
        if path.is_dir():
            return path
    candidates = sorted(runs_dir.glob(f"{profile}_*"))
    return candidates[-1] if candidates else None


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    attacks = [
        json.loads(line)
        for line in (run_dir / "attacks.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    benign_file = run_dir / "benign.jsonl"
    benign = (
        [json.loads(line) for line in benign_file.read_text(encoding="utf-8").splitlines() if line]
        if benign_file.is_file()
        else []
    )
    return summary, attacks, benign
