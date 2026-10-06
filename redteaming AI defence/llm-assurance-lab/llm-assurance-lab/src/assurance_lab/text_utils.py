"""Small text helpers shared by detectors and guardrails."""

from __future__ import annotations

import re

_URL_RE = re.compile(
    r"https?://[^\s)\"'<>]+"
    r"|\b(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:net|com|org|io|info|biz|xyz|example)"
    r"(?:/[^\s)\"'<>]*)?",
    re.IGNORECASE,
)

# Numbers that make a factual claim: percentages and money amounts.
_PERCENT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s?%")
_MONEY_RE = re.compile(
    r"(?:€|eur(?:o|os)?\s?)\s?(\d[\d.,]*)|(\d[\d.,]*)\s?(?:€|eur\b|euro\b|euros\b)",
    re.IGNORECASE,
)


def normalize_number(raw: str) -> str:
    """Normalize '1,75', '1.75', '2,500.00' and '2.500' to a comparable canonical form."""
    s = raw.strip().rstrip(".,")
    if not s:
        return s
    if "," in s and "." in s:
        # The last separator is the decimal separator.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        s = s.replace(",", "") if len(tail) == 3 else head.replace(",", "") + "." + tail
    elif "." in s:
        head, _, tail = s.rpartition(".")
        if len(tail) == 3 and head.isdigit():
            s = s.replace(".", "")
    try:
        value = float(s)
    except ValueError:
        return s
    return f"{value:.2f}".rstrip("0").rstrip(".")


def numeric_claims(text: str) -> set[str]:
    """Return normalized numbers that appear as percentages or money amounts."""
    claims: set[str] = set()
    for m in _PERCENT_RE.finditer(text):
        claims.add(normalize_number(m.group(1)))
    for m in _MONEY_RE.finditer(text):
        raw = m.group(1) or m.group(2)
        if raw:
            claims.add(normalize_number(raw))
    return {c for c in claims if c}


def all_numbers(text: str) -> set[str]:
    return {normalize_number(m) for m in re.findall(r"\d[\d.,]*\d|\d", text)}


def unsupported_numeric_claims(answer: str, context: str) -> set[str]:
    """Numeric claims in ``answer`` that do not occur anywhere in ``context``."""
    return numeric_claims(answer) - all_numbers(context)


def find_urls(text: str) -> list[str]:
    """Find URLs and bare domains. The domain part of an email address is skipped."""
    urls = []
    for m in _URL_RE.finditer(text):
        if m.start() > 0 and text[m.start() - 1] == "@":
            continue
        urls.append(m.group(0).rstrip(".,;:"))
    return urls


def domain_of(url: str) -> str:
    host = re.sub(r"^https?://", "", url, flags=re.IGNORECASE).split("/")[0].lower()
    return host.removeprefix("www.")


def is_allowed_domain(url: str, allowed: list[str]) -> bool:
    host = domain_of(url)
    return any(host == d or host.endswith("." + d) for d in allowed)


def contains_email(text: str) -> bool:
    return re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text) is not None
