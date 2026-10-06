"""HTTP API and the auditor workbench page. Start with ``ledgerlens serve``."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ledgerlens import __version__
from ledgerlens.config import AppConfig, load_config
from ledgerlens.copilot.agent import Copilot
from ledgerlens.copilot.tools import LedgerTools
from ledgerlens.data.chart import account_name
from ledgerlens.data.lines import entry_lines
from ledgerlens.data.schemes import SCHEMES
from ledgerlens.llm.base import LLMError
from ledgerlens.llm.factory import create_llm
from ledgerlens.ml.explain import WEEKDAYS
from ledgerlens.pipeline import load_findings
from ledgerlens.workspace import Workspace

logger = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


class Decision(BaseModel):
    decision: Literal["finding", "dismissed"]
    note: str = Field(default="", max_length=2000)


class State:
    """Loads the workspace once, on first use, and keeps findings and decisions."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._ws: Workspace | None = None
        self._lock = threading.Lock()
        self.findings: dict[str, dict[str, Any]] = {}
        self.decisions: dict[str, dict[str, Any]] = {}

    @property
    def ws(self) -> Workspace:
        with self._lock:
            if self._ws is None:
                try:
                    self._ws = Workspace.load(self.cfg)
                except FileNotFoundError as exc:
                    raise HTTPException(503, f"{exc}. Run ledgerlens all first.") from exc
                self.findings = {f["entry_id"]: f for f in load_findings(self.cfg)}
                path = self.cfg.out / "decisions.jsonl"
                if path.is_file():
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            d = json.loads(line)
                            self.decisions[d["entry_id"]] = d
            return self._ws

    def save_finding(self, inv: dict[str, Any]) -> None:
        self.findings[inv["entry_id"]] = inv
        with (self.cfg.out / "findings.jsonl").open("w", encoding="utf-8") as fh:
            for f in self.findings.values():
                fh.write(json.dumps(f, ensure_ascii=False, default=str) + "\n")

    def save_decision(self, d: dict[str, Any]) -> None:
        self.decisions[d["entry_id"]] = d
        with (self.cfg.out / "decisions.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")


def create_app(cfg: AppConfig | None = None) -> FastAPI:
    state = State(cfg or load_config())
    app = FastAPI(title="LedgerLens", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/summary")
    def summary() -> dict[str, Any]:
        ws = state.ws
        return {
            "company": ws.config.company,
            "financial_year": ws.config.financial_year,
            "entries": len(ws.ledger.entries),
            "flagged_by_rules": int((ws.scores["n_rules"] >= 1).sum()) if ws.scores is not None else None,
            "model": f"{ws.config.llm.provider}:{ws.config.llm.model}",
        }

    @app.get("/api/queue")
    def queue(
        sort: Literal["ledgerlens", "rules"] = "ledgerlens", limit: int = Query(100, ge=1, le=500)
    ) -> list[dict[str, Any]]:
        ws = state.ws
        assert ws.scores is not None
        col = "score_ledgerlens" if sort == "ledgerlens" else "score_rules"
        top = ws.scores.nlargest(limit, col)
        out = []
        for pos, eid in enumerate(top["entry_id"], start=1):
            e = ws.entry(eid)
            reasons = ws.reasons(eid, top=1)
            out.append(
                {
                    "position": pos,
                    "entry_id": eid,
                    "amount": float(e["amount"]),
                    "posting_date": e["posting_date"],
                    "description": e["description"],
                    "top_reason": reasons[0].text if reasons else "",
                    "decision": state.decisions.get(eid, {}).get("decision"),
                }
            )
        return out

    @app.get("/api/entries/{entry_id}")
    def entry(entry_id: str) -> dict[str, Any]:
        ws = state.ws
        try:
            e = ws.entry(entry_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc).strip("'\"")) from exc
        f = ws.feature_row(entry_id)
        vendor, customer = str(e["vendor_id"]), str(e["customer_id"])
        counterparty = ws.vendor_names.get(vendor) or ws.customer_names.get(customer) or ""
        return {
            "entry_id": entry_id,
            "amount": float(e["amount"]),
            "entry_type": e["entry_type"],
            "posting_date": e["posting_date"],
            "posting_time": e["posting_time"],
            "weekday": WEEKDAYS[int(f["weekday"])],
            "created_by": ws.user_names.get(str(e["created_by"]), e["created_by"]),
            "approved_by": ws.user_names.get(str(e["approved_by"]), e["approved_by"]),
            "description": e["description"],
            "counterparty": counterparty,
            "accounts": f"{account_name(e['dr_account'])} / {account_name(e['cr_account'])}",
            "lines": entry_lines(e),
            "reasons": [r.as_dict() for r in ws.reasons(entry_id)],
            "rules": ws.rule_hits(entry_id),
            "related": LedgerTools(ws).related_entries(entry_id)["related"],
            "rank": int(ws.score_row(entry_id).get("rank", 0)),
            "investigation": state.findings.get(entry_id),
            "decision": state.decisions.get(entry_id),
        }

    @app.post("/api/entries/{entry_id}/investigate")
    def investigate(entry_id: str) -> dict[str, Any]:
        ws = state.ws
        try:
            inv = Copilot(ws, create_llm(ws.config.llm), ws.config.copilot.max_steps).investigate(entry_id).as_dict()
        except KeyError as exc:
            raise HTTPException(404, str(exc).strip("'\"")) from exc
        except LLMError as exc:
            raise HTTPException(502, str(exc)) from exc
        state.save_finding(inv)
        return inv

    @app.post("/api/entries/{entry_id}/decision")
    def decide(entry_id: str, body: Decision) -> dict[str, Any]:
        ws = state.ws
        try:
            ws.position(entry_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc).strip("'\"")) from exc
        record = {
            "entry_id": entry_id,
            **body.model_dump(),
            "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        state.save_decision(record)
        return record

    @app.get("/api/evaluation")
    def evaluation() -> dict[str, Any]:
        path = state.cfg.out / "metrics.json"
        if not path.is_file():
            raise HTTPException(503, "No metrics yet. Run ledgerlens analyze first.")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        metrics["scheme_titles"] = {k: v["title"] for k, v in SCHEMES.items()}
        return metrics

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


app = create_app()
