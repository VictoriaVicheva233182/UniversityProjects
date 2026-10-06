"""Deterministic output controls applied after the model answers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from assurance_lab.rag.customers import Customer
from assurance_lab.text_utils import find_urls, is_allowed_domain, unsupported_numeric_claims

BLOCKED_ANSWER = (
    "I can't share that. I can help with questions about Havenkade products and your own account."
)
UNGROUNDED_ANSWER = (
    "I don't have verified information about that. Please check havenkade.example or contact us "
    "via havenkade.example/contact."
)
CREDENTIAL_WARNING = (
    "For your security: Havenkade will never ask for your PIN, password or security code. "
    "Please only use the official app and havenkade.example."
)

_CREDENTIAL_REQUEST = re.compile(
    r"\b(verify|confirm|provide|share|send|give|type in)\b[^.]{0,60}\b(pin|password|security code|login details)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"\b(never|not|don't|do not|won't)\b", re.IGNORECASE)


@dataclass
class FilterResult:
    text: str
    blocked: bool = False
    actions: list[str] = field(default_factory=list)


class OutputFilter:
    def __init__(
        self,
        canaries: Iterable[str],
        customers: Iterable[Customer],
        allowed_domains: list[str],
        grounding_check: bool = True,
    ) -> None:
        self.canaries = [c for c in canaries if c]
        self.customers = list(customers)
        self.allowed_domains = allowed_domains
        self.grounding_check = grounding_check

    def apply(self, answer: str, session_customer_id: str, context: str) -> FilterResult:
        result = FilterResult(text=answer)

        lowered = answer.lower()
        if any(c.lower() in lowered for c in self.canaries):
            return FilterResult(BLOCKED_ANSWER, blocked=True, actions=["canary_leak_blocked"])

        for sentence in re.split(r"(?<=[.!?])\s+", answer):
            if _CREDENTIAL_REQUEST.search(sentence) and not _NEGATION.search(sentence):
                return FilterResult(
                    CREDENTIAL_WARNING, blocked=True, actions=["credential_request_blocked"]
                )

        text = answer
        for customer in self.customers:
            if customer.customer_id == session_customer_id:
                continue
            for marker in customer.pii_markers():
                if marker and marker in text:
                    text = text.replace(marker, "[REDACTED]")
                    if "foreign_pii_redacted" not in result.actions:
                        result.actions.append("foreign_pii_redacted")

        for url in find_urls(text):
            if not is_allowed_domain(url, self.allowed_domains):
                text = text.replace(url, "[link removed]")
                if "external_link_removed" not in result.actions:
                    result.actions.append("external_link_removed")

        if self.grounding_check and unsupported_numeric_claims(text, context):
            return FilterResult(
                UNGROUNDED_ANSWER,
                blocked=True,
                actions=[*result.actions, "ungrounded_numbers_replaced"],
            )

        result.text = text
        return result
