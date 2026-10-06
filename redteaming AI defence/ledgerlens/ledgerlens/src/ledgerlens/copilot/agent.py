"""The investigating copilot.

It works like a junior auditor with read-only access to the ledger: it calls tools
to gather evidence about one flagged entry, then writes a draft finding. The draft
is checked against the evidence before an auditor sees it.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ledgerlens.copilot.grounding import Grounding, check_grounding
from ledgerlens.copilot.tools import TOOL_DESCRIPTIONS, LedgerTools, ToolError
from ledgerlens.data.schemes import SCHEMES
from ledgerlens.llm.base import LLMClient, LLMError, Message
from ledgerlens.workspace import Workspace

logger = logging.getLogger(__name__)

SCHEME_CHOICES = [*SCHEMES, "none"]


class Finding(BaseModel):
    assessment: Literal["suspicious", "likely_legitimate", "needs_more_evidence"] = "needs_more_evidence"
    suspected_scheme: str = "none"
    title: str = "Entry needs review"
    summary: str = ""
    evidence: list[str] = Field(default_factory=list)
    risk: str = ""
    next_steps: list[str] = Field(default_factory=list)

    @field_validator("assessment", mode="before")
    @classmethod
    def _assessment(cls, v: object) -> str:
        v = str(v).strip().lower().replace(" ", "_")
        return v if v in ("suspicious", "likely_legitimate", "needs_more_evidence") else "needs_more_evidence"

    @field_validator("suspected_scheme", mode="before")
    @classmethod
    def _scheme(cls, v: object) -> str:
        v = str(v).strip().lower()
        return v if v in SCHEME_CHOICES else "none"

    @field_validator("evidence", "next_steps", mode="before")
    @classmethod
    def _list(cls, v: object) -> list[str]:
        if isinstance(v, str):
            return [v]
        return [str(x) for x in v] if isinstance(v, list) else []

    def claims_text(self) -> str:
        return " ".join([self.title, self.summary, *self.evidence, self.risk])


@dataclass
class Step:
    tool: str
    args: dict[str, Any]
    ok: bool
    summary: str


@dataclass
class Investigation:
    entry_id: str
    finding: dict[str, Any]
    grounding: dict[str, Any]
    steps: list[Step] = field(default_factory=list)
    fallback_used: bool = False
    model: str = ""
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["steps"] = [asdict(s) for s in self.steps]
        return d


def system_prompt(company: str, year: int, max_steps: int) -> str:
    tools = "\n".join(f"- {t}" for t in TOOL_DESCRIPTIONS.values())
    return f"""You are an audit assistant. You help an external auditor test journal entries for fraud risk (ISA 240) at {company}, financial year {year}.
You investigate ONE flagged journal entry by calling tools, then you write a draft finding for the auditor to review.

Tools:
{tools}

Reply with ONE JSON object and nothing else. Either call a tool:
{{"action": "tool", "tool": "<tool name>", "args": {{"<argument>": "<value>"}}}}
or, when you have enough evidence (at most {max_steps} tool calls), finish:
{{"action": "finish", "finding": {{"assessment": "suspicious" or "likely_legitimate" or "needs_more_evidence", "suspected_scheme": one of {json.dumps(SCHEME_CHOICES)}, "title": "short title", "summary": "two or three plain sentences", "evidence": ["one fact per item, with exact numbers from the tool results"], "risk": "why this matters for the financial statements", "next_steps": ["audit procedures to perform next"]}}}}

Rules:
- Only state facts that appear in tool results. Copy amounts, dates, times and IDs exactly.
- Never invent documents, emails or interviews. If something needs checking, put it in next_steps.
- Look at the user's normal behaviour and related entries before you decide.
- This is a draft for a human auditor. Do not conclude that fraud happened; describe the risk."""


def parse_json(text: str) -> dict[str, Any] | None:
    """Take the first JSON object from a model reply, tolerating code fences and chatter."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _fallback_finding(tools: LedgerTools, entry_id: str) -> Finding:
    entry = tools.get_entry(entry_id)
    return Finding(
        assessment="needs_more_evidence",
        title=f"Review {entry_id}: flagged by the model",
        summary="The copilot could not complete its investigation. These are the risk reasons found by the detection step.",
        evidence=entry["risk_reasons"],
        next_steps=[
            "Inspect the supporting documents for this entry.",
            "Discuss the entry with the person who posted it.",
        ],
    )


class Copilot:
    def __init__(self, ws: Workspace, llm: LLMClient, max_steps: int = 6) -> None:
        self.ws = ws
        self.llm = llm
        self.tools = LedgerTools(ws)
        self.max_steps = max_steps
        self.system = system_prompt(ws.config.company, ws.config.financial_year, max_steps)

    def investigate(self, entry_id: str) -> Investigation:
        start = time.perf_counter()
        self.ws.position(entry_id)  # raises KeyError for unknown entries
        messages = [Message("system", self.system), Message("user", f"Investigate journal entry {entry_id}.")]
        evidence_parts = [self.system, entry_id]
        steps: list[Step] = []
        finding: Finding | None = None
        invalid = 0
        for _ in range(self.max_steps + 4):
            reply = self.llm.chat(messages, json_mode=True)
            obj = parse_json(reply)
            messages.append(Message("assistant", reply))
            if obj is None or obj.get("action") not in ("tool", "finish"):
                invalid += 1
                if invalid >= 2:
                    break
                messages.append(
                    Message("user", 'Reply with one JSON object only, with "action" set to "tool" or "finish".')
                )
                continue
            if obj["action"] == "finish":
                try:
                    finding = Finding.model_validate(obj.get("finding") or {})
                except ValueError:
                    finding = None
                break
            if len(steps) >= self.max_steps:
                messages.append(Message("user", "You have used all tool calls. Finish now with your finding."))
                continue
            name, args = str(obj.get("tool", "")), obj.get("args") or {}
            try:
                result = self.tools.call(name, args if isinstance(args, dict) else {})
                text = json.dumps(result, ensure_ascii=False, default=str)
                evidence_parts.append(text)
                steps.append(Step(name, args, True, _summarize(name, result)))
                messages.append(Message("user", f"Result of {name}: {text}"))
            except ToolError as exc:
                steps.append(Step(name, args if isinstance(args, dict) else {}, False, str(exc)))
                messages.append(Message("user", f"Error from {name}: {exc}"))

        fallback = finding is None
        if finding is None:
            finding = _fallback_finding(self.tools, entry_id)
            evidence_parts.append(json.dumps(self.tools.get_entry(entry_id), default=str))
        grounding: Grounding = check_grounding(finding.claims_text(), "\n".join(evidence_parts))
        return Investigation(
            entry_id=entry_id,
            finding=finding.model_dump(),
            grounding=asdict(grounding),
            steps=steps,
            fallback_used=fallback,
            model=self.llm.describe(),
            seconds=round(time.perf_counter() - start, 1),
        )


def _summarize(name: str, result: dict[str, Any]) -> str:
    if name == "get_entry":
        return f"{result['type']} of €{result['amount_eur']:,.2f} by {result['created_by']}"
    if name == "user_profile":
        return f"{result['name']}, {result['role']}, usually posts {result['usual_posting_hours']}"
    if name == "vendor_profile":
        return f"{result['name']}, created {result['created_date']} by {result['created_by']}"
    if name == "related_entries":
        return f"{result['total_found']} related entries"
    if name == "account_pair":
        return f"{result['names']} used {result['entries_this_year']} times"
    return "done"


def investigate_top(ws: Workspace, llm: LLMClient, top_n: int, max_steps: int) -> list[Investigation]:
    if ws.scores is None:
        raise ValueError("Scores are needed to pick the top entries. Run: ledgerlens analyze")
    queue = ws.scores.sort_values("rank").head(top_n)["entry_id"].tolist()
    copilot = Copilot(ws, llm, max_steps)
    out = []
    for entry_id in queue:
        try:
            out.append(copilot.investigate(entry_id))
        except LLMError:
            raise
        except Exception:
            logger.exception("Investigation of %s failed", entry_id)
    return out
