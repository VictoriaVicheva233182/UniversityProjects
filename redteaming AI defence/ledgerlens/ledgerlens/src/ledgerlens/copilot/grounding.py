"""Check that every number and identifier in a finding comes from the evidence.

The copilot may only state facts it got from tools. Anything it writes that is not in
the tool results is marked, so the auditor sees exactly which claims to verify.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUM_RE = re.compile(r"\d[\d.,]*\d|\d")
_ID_RE = re.compile(r"\b(JE\d{6}|V\d{4}|U\d{3}|C\d{4})\b")


def normalize_number(raw: str) -> str:
    s = raw.strip().rstrip(".,")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        s = s.replace(",", "") if len(tail) == 3 else head.replace(",", "") + "." + tail
    elif "." in s:
        head, _, tail = s.rpartition(".")
        if len(tail) == 3 and head.isdigit() and s.count(".") >= 1 and len(head) <= 3:
            s = s.replace(".", "")
    try:
        value = float(s)
    except ValueError:
        return s
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _tokens(text: str) -> set[str]:
    tokens = set()
    for raw in re.findall(r"[\d][\d.,:\-/]*\d|\d", text):
        tokens.add(normalize_number(raw))
        for part in re.split(r"[-:/]", raw):
            if part:
                tokens.add(normalize_number(part))
    return tokens


@dataclass
class Grounding:
    grounded: bool
    ungrounded_numbers: list[str] = field(default_factory=list)
    unknown_ids: list[str] = field(default_factory=list)


def check_grounding(claims: str, evidence: str) -> Grounding:
    allowed = _tokens(evidence)
    claimed = {normalize_number(t) for t in re.findall(r"\d[\d.,]*\d|\d", claims)}
    bad = sorted(n for n in claimed if n not in allowed)
    ids = sorted({i for i in _ID_RE.findall(claims) if i not in evidence})
    return Grounding(grounded=not bad and not ids, ungrounded_numbers=bad, unknown_ids=ids)
