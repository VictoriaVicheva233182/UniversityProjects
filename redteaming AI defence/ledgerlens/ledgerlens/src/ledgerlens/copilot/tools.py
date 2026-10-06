"""Read-only ledger tools the copilot can call. Every result is plain JSON data."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ledgerlens.data.chart import account_name
from ledgerlens.data.lines import entry_lines
from ledgerlens.ml.explain import WEEKDAYS, hhmm
from ledgerlens.workspace import Workspace

TOOL_DESCRIPTIONS = {
    "get_entry": "get_entry(entry_id): the entry, its lines, rule hits and risk reasons",
    "user_profile": "user_profile(user_id): how this user normally works",
    "vendor_profile": "vendor_profile(vendor_id): the supplier's history",
    "related_entries": "related_entries(entry_id): linked entries (same supplier, same person same day, reversals)",
    "account_pair": "account_pair(dr_account, cr_account): how often this account combination is used",
}


class ToolError(ValueError):
    pass


class LedgerTools:
    def __init__(self, ws: Workspace) -> None:
        self.ws = ws
        self.e = ws.ledger.entries

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        tools: dict[str, Callable[..., dict[str, Any]]] = {
            "get_entry": self.get_entry,
            "user_profile": self.user_profile,
            "vendor_profile": self.vendor_profile,
            "related_entries": self.related_entries,
            "account_pair": self.account_pair,
        }
        if name not in tools:
            raise ToolError(f"Unknown tool '{name}'. Available: {', '.join(tools)}")
        try:
            return tools[name](**{k: str(v) for k, v in args.items()})
        except TypeError as exc:
            raise ToolError(f"Wrong arguments for {name}: {exc}") from exc
        except KeyError as exc:
            raise ToolError(str(exc).strip("'\"")) from exc

    def _person(self, user_id: str) -> str:
        return f"{self.ws.user_names.get(user_id, user_id)} ({user_id})" if user_id else ""

    def _brief(self, row: pd.Series, relation: str) -> dict[str, Any]:
        return {
            "entry_id": row["entry_id"], "date": row["posting_date"], "time": row["posting_time"],
            "type": row["entry_type"], "amount_eur": round(float(row["amount"]), 2),
            "created_by": row["created_by"], "description": row["description"], "relation": relation,
        }  # fmt: skip

    def get_entry(self, entry_id: str) -> dict[str, Any]:
        e = self.ws.entry(entry_id)
        f = self.ws.feature_row(entry_id)
        reasons = self.ws.reasons(entry_id, top=6)
        vendor = str(e["vendor_id"])
        customer = str(e["customer_id"])
        score = self.ws.score_row(entry_id)
        return {
            "entry_id": entry_id,
            "date": e["posting_date"],
            "weekday": WEEKDAYS[int(f["weekday"])],
            "time": e["posting_time"],
            "type": e["entry_type"],
            "source": e["source"],
            "created_by": self._person(str(e["created_by"])),
            "approved_by": self._person(str(e["approved_by"])) or "no approver",
            "amount_eur": round(float(e["amount"]), 2),
            "lines": entry_lines(e),
            "vendor": f"{self.ws.vendor_names.get(vendor, vendor)} ({vendor})" if vendor else None,
            "customer": f"{self.ws.customer_names.get(customer, customer)} ({customer})" if customer else None,
            "reference": e["reference"] or None,
            "description": e["description"],
            "rule_hits": self.ws.rule_hits(entry_id),
            "risk_reasons": [r.text for r in reasons],
            "reason_codes": [r.code for r in reasons],
            "review_rank": int(score["rank"]) if score else None,
        }

    def user_profile(self, user_id: str) -> dict[str, Any]:
        users = self.ws.ledger.users.set_index("user_id")
        if user_id not in users.index:
            raise ToolError(f"Unknown user: {user_id}")
        mask = (self.e["created_by"] == user_id).to_numpy()
        f = self.ws.features[mask]
        pairs = f["pair"].value_counts().head(3)
        return {
            "user_id": user_id,
            "name": users.loc[user_id, "name"],
            "role": users.loc[user_id, "role"],
            "can_approve": bool(users.loc[user_id, "can_approve"]),
            "entries_this_year": int(mask.sum()),
            "manual_entries": int(f["is_manual"].sum()),
            "usual_posting_hours": f"{hhmm(float(f['hour'].quantile(0.1)))} to {hhmm(float(f['hour'].quantile(0.9)))}",
            "entries_after_20h_or_before_7h": int(f["off_hours"].sum()),
            "weekend_entries": int(f["weekend_or_holiday"].sum()),
            "most_used_account_combinations": [
                {"accounts": p, "names": " / ".join(account_name(a) for a in p.split("/")), "entries": int(n)}
                for p, n in pairs.items()
            ],
            "suppliers_created": int((self.ws.ledger.vendors["created_by"] == user_id).sum()),
        }

    def vendor_profile(self, vendor_id: str) -> dict[str, Any]:
        vendors = self.ws.ledger.vendors.set_index("vendor_id")
        if vendor_id not in vendors.index:
            raise ToolError(f"Unknown supplier: {vendor_id}")
        v = vendors.loc[vendor_id]
        rows = self.e[self.e["vendor_id"] == vendor_id]
        inv = rows[rows["entry_type"] == "purchase_invoice"]
        pay = rows[rows["entry_type"].isin(["vendor_payment", "manual_payment"])]
        round_share = round(100 * float(((inv["amount"] % 500) == 0).mean())) if len(inv) else 0
        return {
            "vendor_id": vendor_id,
            "name": v["name"],
            "category": v["category"],
            "created_date": v["created_date"],
            "created_by": self._person(str(v["created_by"])),
            "invoices": len(inv),
            "invoiced_total_eur": round(float(inv["amount"].sum()), 2),
            "first_invoice": inv["posting_date"].min() if len(inv) else None,
            "last_invoice": inv["posting_date"].max() if len(inv) else None,
            "payments": len(pay),
            "paid_total_eur": round(float(pay["amount"].sum()), 2),
            "manual_payments": int((pay["entry_type"] == "manual_payment").sum()),
            "round_invoice_share_pct": round_share,
            "booked_by": sorted({self._person(u) for u in rows["created_by"]}),
        }

    def related_entries(self, entry_id: str) -> dict[str, Any]:
        e = self.ws.entry(entry_id)
        d = pd.Timestamp(e["posting_date"])
        dates = pd.to_datetime(self.e["posting_date"])
        near = ((dates - d).abs().dt.days <= 30).to_numpy()
        others = (self.e["entry_id"] != entry_id).to_numpy()
        out: list[dict[str, Any]] = []
        same_day = (
            others
            & (self.e["created_by"] == e["created_by"]).to_numpy()
            & (self.e["posting_date"] == e["posting_date"]).to_numpy()
        )
        if str(e["created_by"]).startswith("SYS"):
            same_day &= False
        for _, row in self.e[same_day].head(5).iterrows():
            out.append(self._brief(row, "same person, same day"))
        swapped = (
            others
            & near
            & (self.e["dr_account"] == e["cr_account"]).to_numpy()
            & (self.e["cr_account"] == e["dr_account"]).to_numpy()
        )
        swapped &= np.isclose(self.e["amount"].astype(float), float(e["amount"]))
        for _, row in self.e[swapped].head(3).iterrows():
            out.append(self._brief(row, "possible reversal: same amount, accounts swapped"))
        if e["vendor_id"]:
            vend = others & near & (self.e["vendor_id"] == e["vendor_id"]).to_numpy()
            for _, row in self.e[vend].head(6).iterrows():
                out.append(self._brief(row, "same supplier within 30 days"))
        seen: set[str] = set()
        unique = [r for r in out if not (r["entry_id"] in seen or seen.add(r["entry_id"]))]  # type: ignore[func-returns-value]
        return {"entry_id": entry_id, "related": unique[:10], "total_found": len(unique)}

    def account_pair(self, dr_account: str, cr_account: str) -> dict[str, Any]:
        mask = ((self.e["dr_account"] == dr_account) & (self.e["cr_account"] == cr_account)).to_numpy()
        rows = self.e[mask]
        by_user = rows["created_by"].value_counts().head(5)
        return {
            "accounts": f"{dr_account}/{cr_account}",
            "names": f"{account_name(dr_account)} / {account_name(cr_account)}",
            "entries_this_year": int(mask.sum()),
            "median_amount_eur": round(float(rows["amount"].median()), 2) if len(rows) else None,
            "posted_by": [{"user": self._person(u), "entries": int(n)} for u, n in by_user.items()],
        }
