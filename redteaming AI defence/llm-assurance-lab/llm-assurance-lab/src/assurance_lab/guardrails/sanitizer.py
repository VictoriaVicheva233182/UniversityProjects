"""Remove instruction-like sentences from untrusted retrieved text.

This is a deterministic first line of defense against indirect prompt injection.
It is deliberately conservative: it only touches documents marked ``untrusted``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_INJECTION_PATTERNS = [
    r"\bignore\b.{0,40}\binstructions?\b",
    r"\bdisregard\b.{0,40}\b(instructions?|rules?)\b",
    r"\b(system|important)\s+note\s+for\s+ai\b",
    r"\b(ai|llm)\s+(assistants?|systems?|models?)\b.{0,80}\b(must|should|tell|say|inform)\b",
    r"\byou\s+must\s+(now\s+)?(tell|say|inform|instruct)\b",
    r"\bverify\b.{0,60}\b(pin|password|security code)\b",
    r"\bnew\s+instructions?\b",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE | re.DOTALL)


@dataclass
class SanitizeResult:
    text: str
    removed: list[str]


def sanitize_untrusted(text: str) -> SanitizeResult:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept: list[str] = []
    removed: list[str] = []
    for sentence in sentences:
        (removed if _INJECTION_RE.search(sentence) else kept).append(sentence)
    return SanitizeResult(" ".join(kept).strip(), removed)


def looks_like_injection(text: str) -> bool:
    return _INJECTION_RE.search(text) is not None
