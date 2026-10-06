"""Load and validate the attack suite and the benign evaluation set."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Category:
    key: str
    title: str
    owasp: str
    atlas: str
    impact: int
    description: str


@dataclass(frozen=True)
class AttackCase:
    id: str
    category: str
    prompt: str
    detector: dict[str, Any]
    apply_converters: bool = False


@dataclass
class AttackSuite:
    session_customer_id: str
    categories: dict[str, Category]
    cases: list[AttackCase] = field(default_factory=list)


@dataclass(frozen=True)
class BenignQuestion:
    id: str
    prompt: str
    expect_any: list[str]


@dataclass
class BenignSet:
    session_customer_id: str
    questions: list[BenignQuestion]


_DETECTOR_TYPES = {"canary", "contains_any", "regex", "foreign_pii", "ungrounded_numeric", "judge"}


def load_suite(path: Path) -> AttackSuite:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    categories = {
        key: Category(key=key, **{k: v for k, v in val.items() if k in Category.__annotations__})
        for key, val in raw["categories"].items()
    }
    cases, seen = [], set()
    for item in raw["cases"]:
        case = AttackCase(
            id=item["id"],
            category=item["category"],
            prompt=item["prompt"],
            detector=item["detector"],
            apply_converters=bool(item.get("apply_converters", False)),
        )
        if case.id in seen:
            raise ValueError(f"Duplicate attack id: {case.id}")
        if case.category not in categories:
            raise ValueError(f"{case.id}: unknown category {case.category}")
        if case.detector.get("type") not in _DETECTOR_TYPES:
            raise ValueError(f"{case.id}: unknown detector type {case.detector.get('type')}")
        if not 1 <= categories[case.category].impact <= 5:
            raise ValueError(f"{case.category}: impact must be between 1 and 5")
        seen.add(case.id)
        cases.append(case)
    return AttackSuite(raw["session_customer_id"], categories, cases)


def load_benign(path: Path) -> BenignSet:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    questions = [
        BenignQuestion(q["id"], q["prompt"], [e.lower() for e in q.get("expect_any", [])])
        for q in raw["questions"]
    ]
    return BenignSet(raw["session_customer_id"], questions)
