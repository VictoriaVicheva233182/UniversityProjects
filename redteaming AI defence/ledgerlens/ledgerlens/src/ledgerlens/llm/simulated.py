"""A scripted stand-in for an LLM, used for tests, CI and offline demos.

It follows the copilot's JSON protocol: it calls get_entry, user_profile,
vendor_profile (when there is a supplier) and related_entries, then writes a
finding from the tool results. Its findings show the plumbing works. They are
not the judgement of a real model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from ledgerlens.llm.base import LLMClient, Message

_SCHEME_BY_REASON = [
    ("split", "split_payments"),
    ("dup_reference", "duplicate_invoices"),
    ("dup_amount", "duplicate_invoices"),
    ("same_person_vendor", "ghost_vendor"),
    ("self_approved", "self_approval"),
    ("manual_revenue", "fictitious_revenue"),
    ("vague", "misc_cash"),
]


class SimulatedLLM(LLMClient):
    name = "simulated"

    def __init__(self, model: str = "scripted-v1", **_: object) -> None:
        super().__init__(model)

    def chat(self, messages: Sequence[Message], json_mode: bool = False) -> str:
        first_user = next((m.content for m in messages if m.role == "user"), "")
        m = re.search(r"JE\d{6}", first_user)
        if not m:
            return "Reply with the single word: ready" if not json_mode else '{"status": "ready"}'
        entry_id = m.group(0)
        results: dict[str, dict] = {}
        for msg in messages:
            if msg.role == "user" and msg.content.startswith("Result of "):
                name, _, payload = msg.content[len("Result of ") :].partition(": ")
                results[name] = json.loads(payload)
        entry = results.get("get_entry")
        if entry is None:
            return json.dumps({"action": "tool", "tool": "get_entry", "args": {"entry_id": entry_id}})
        user_id = re.search(r"\((\w+-?\w*)\)$", entry["created_by"])
        if "user_profile" not in results and user_id:
            return json.dumps({"action": "tool", "tool": "user_profile", "args": {"user_id": user_id.group(1)}})
        vendor = re.search(r"\((V\d{4})\)", entry.get("vendor") or "")
        if vendor and "vendor_profile" not in results:
            return json.dumps({"action": "tool", "tool": "vendor_profile", "args": {"vendor_id": vendor.group(1)}})
        if "related_entries" not in results:
            return json.dumps({"action": "tool", "tool": "related_entries", "args": {"entry_id": entry_id}})

        codes = entry.get("reason_codes", [])
        scheme = next((s for code, s in _SCHEME_BY_REASON if code in codes), "none")
        profile = results.get("user_profile", {})
        evidence = list(entry.get("risk_reasons", []))
        if profile:
            evidence.append(f"{profile['name']} usually posts from {profile['usual_posting_hours']}.")
        finding = {
            "assessment": "suspicious" if len(codes) >= 2 else "needs_more_evidence",
            "suspected_scheme": scheme,
            "title": (entry.get("risk_reasons") or ["Entry needs review"])[0].rstrip("."),
            "summary": f"{entry['type'].replace('_', ' ').capitalize()} of €{entry['amount_eur']:,.2f} posted by {entry['created_by']} on {entry['date']} at {entry['time']}.",
            "evidence": evidence,
            "risk": "If this entry is not supported, the financial statements could be misstated.",
            "next_steps": ["Inspect the supporting documents.", "Ask the person who posted it to explain the entry."],
        }
        return json.dumps({"action": "finish", "finding": finding})
