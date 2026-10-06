"""Expand an entry header into its debit and credit lines."""

from __future__ import annotations

from typing import Any

from ledgerlens.data.chart import account_name


def entry_lines(entry: dict[str, Any]) -> list[dict[str, Any]]:
    amount = round(float(entry["amount"]), 2)
    vat = round(float(entry.get("vat_amount") or 0.0), 2)
    dr, cr = str(entry["dr_account"]), str(entry["cr_account"])
    side = entry.get("vat_side", "")
    if vat > 0 and side == "cr":
        lines = [(dr, amount, 0.0), (cr, 0.0, round(amount - vat, 2)), ("2200", 0.0, vat)]
    elif vat > 0 and side == "dr":
        lines = [(dr, round(amount - vat, 2), 0.0), ("2210", vat, 0.0), (cr, 0.0, amount)]
    else:
        lines = [(dr, amount, 0.0), (cr, 0.0, amount)]
    return [{"account": a, "account_name": account_name(a), "debit": d, "credit": c} for a, d, c in lines]
