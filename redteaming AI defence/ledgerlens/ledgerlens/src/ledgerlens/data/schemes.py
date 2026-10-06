"""Fraud schemes planted in the ledger, modelled on common journal entry fraud patterns.

Each scheme returns the entries it adds, tagged in the hidden ``_scheme`` column.
The tag is split off into ``ground_truth.csv`` and is never used for detection.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from ledgerlens.data.common import BUSINESS_DAYS, YEAR, frame, last_business_day, money, next_business_day, times

SCHEMES: dict[str, dict[str, str]] = {
    "fictitious_revenue": {
        "title": "Fictitious revenue at quarter end",
        "how": "The controller books large round revenue amounts late on the last day of each quarter and reverses them after the quarter closes.",
    },
    "split_payments": {
        "title": "Payments split under the approval limit",
        "how": "An accounts payable clerk pays one supplier several times on the same day, each time just under the 10,000 euro approval limit, without invoices.",
    },
    "ghost_vendor": {
        "title": "Ghost supplier",
        "how": "An accounts payable clerk creates a new supplier, then books and pays its round monthly invoices herself.",
    },
    "self_approval": {
        "title": "Self-approved cash journals",
        "how": "A general ledger accountant approves his own large journals that take money straight out of the bank account.",
    },
    "night_postings": {
        "title": "Night-time postings by the payroll officer",
        "how": "The payroll officer moves money from payroll liabilities to the bank account at night, at the weekend.",
    },
    "expense_capitalisation": {
        "title": "Expenses moved to fixed assets at year end",
        "how": "At year end the controller reclassifies operating costs as fixed assets, which inflates profit.",
    },
    "duplicate_invoices": {
        "title": "Duplicate supplier invoices",
        "how": "Existing supplier invoices are booked a second time a few days later with the same amount; half of them with a slightly changed reference.",
    },
    "misc_cash": {
        "title": "Cash booked to miscellaneous expenses",
        "how": "Small cash withdrawals are booked to miscellaneous expenses with vague descriptions.",
    },
}


def _gl(rng: np.random.Generator, n: int, scheme: str, **cols: object) -> pd.DataFrame:
    cols.setdefault("source", "GL")
    cols.setdefault("entry_type", "manual_journal")
    cols.setdefault("approved_by", "")
    return frame(_scheme=[scheme] * n, **cols)


def fictitious_revenue(rng: np.random.Generator, customers: pd.DataFrame) -> pd.DataFrame:
    big = customers.sort_values("weight", ascending=False)["customer_id"].head(5).tolist()
    amounts = [48_000.0, 65_000.0, 72_500.0, 90_000.0]
    rows = []
    for q, amount in enumerate(amounts, start=1):
        cust = big[q % len(big)]
        d = last_business_day(YEAR, q * 3)
        rows.append((d, times(rng, 1, 21.5, 22.8)[0], "1100", "4000", amount, cust, f"Revenue adjustment Q{q}"))
        if q < 4:
            rd = next_business_day(date(YEAR, q * 3 + 1, 3))
            rows.append(
                (rd, times(rng, 1, 9.0, 10.0)[0], "4000", "1100", amount, cust, f"Reversal revenue adjustment Q{q}")
            )
    return _gl(
        rng, len(rows), "fictitious_revenue",
        posting_date=[r[0].isoformat() for r in rows], posting_time=[r[1] for r in rows],
        created_by="U107", approved_by="U108", dr_account=[r[2] for r in rows], cr_account=[r[3] for r in rows],
        amount=[r[4] for r in rows], customer_id=[r[5] for r in rows], description=[r[6] for r in rows],
    )  # fmt: skip


def split_payments(rng: np.random.Generator, vendors: pd.DataFrame, limit: float) -> pd.DataFrame:
    old = vendors[(vendors["category"] == "haulage subcontractor") & (vendors["created_date"] < "2024-01-01")]
    v = old.iloc[3]
    rows = []
    for month, count in ((4, 3), (7, 4), (10, 3)):
        d = next_business_day(date(YEAR, month, 15))
        start = 16.0
        for i in range(count):
            t = times(rng, 1, start + i * 0.15, start + i * 0.15 + 0.1)[0]
            amount = float(money(rng.uniform(limit * 0.962, limit * 0.998)))
            rows.append((d, t, amount, f"{v['vendor_id']}-ADV{month:02d}{i + 1}"))
    return _gl(
        rng, len(rows), "split_payments",
        posting_date=[r[0].isoformat() for r in rows], posting_time=[r[1] for r in rows],
        source="BANK-MANUAL", entry_type="manual_payment", created_by="U102",
        dr_account="2000", cr_account="1000", amount=[r[2] for r in rows], vendor_id=v["vendor_id"],
        reference=[r[3] for r in rows], description=[f"Advance payment {v['name']}"] * len(rows),
    )  # fmt: skip


def ghost_vendor(rng: np.random.Generator, vendors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    vid = f"V{len(vendors) + 1:04d}"
    vendor = {
        "vendor_id": vid, "name": "Van Dijk Advies", "category": "consulting", "account": "6400",
        "pattern": "variable", "mu": 0.0, "sigma": 0.0, "fixed_amount": 0.0, "vat_rate": 0.0,
        "weight": 0.0, "created_date": f"{YEAR}-05-12", "created_by": "U101",
    }  # fmt: skip
    amounts = [6500.0, 7500.0, 8000.0, 8500.0, 9000.0, 7500.0, 9500.0]
    rows = []
    for i, month in enumerate(range(6, 13)):
        d = next_business_day(date(YEAR, month, 10))
        ref = f"VDA-2025-{i + 1:03d}"
        rows.append(
            (
                d,
                times(rng, 1, 15.0, 17.0)[0],
                "purchase_invoice",
                "AP",
                "6400",
                "2000",
                amounts[i],
                ref,
                f"Invoice {ref} Van Dijk Advies",
            )
        )
        pd_ = next_business_day(d + timedelta(days=int(rng.integers(2, 5))))
        rows.append(
            (
                pd_,
                times(rng, 1, 16.0, 17.5)[0],
                "manual_payment",
                "BANK-MANUAL",
                "2000",
                "1000",
                amounts[i],
                ref,
                f"Payment {ref}",
            )
        )
    entries = _gl(
        rng, len(rows), "ghost_vendor",
        posting_date=[r[0].isoformat() for r in rows], posting_time=[r[1] for r in rows],
        entry_type=[r[2] for r in rows], source=[r[3] for r in rows], created_by="U101",
        dr_account=[r[4] for r in rows], cr_account=[r[5] for r in rows], amount=[r[6] for r in rows],
        vat_side=["dr" if r[2] == "purchase_invoice" else "" for r in rows], vendor_id=vid,
        reference=[r[7] for r in rows], description=[r[8] for r in rows],
    )  # fmt: skip
    return entries, pd.concat([vendors, pd.DataFrame([vendor])], ignore_index=True)


def self_approval(rng: np.random.Generator) -> pd.DataFrame:
    n = 6
    days = sorted(rng.choice(len(BUSINESS_DAYS), n, replace=False))
    return _gl(
        rng, n, "self_approval",
        posting_date=[BUSINESS_DAYS[i].isoformat() for i in days], posting_time=times(rng, n, 11.0, 16.0),
        created_by="U104", approved_by="U104", dr_account="6400", cr_account="1000",
        amount=money(rng.uniform(12_000, 28_000, n)), description="Consultancy fees settlement",
    )  # fmt: skip


def night_postings(rng: np.random.Generator) -> pd.DataFrame:
    weekend = [d for d in (date(YEAR, 1, 1) + timedelta(days=i) for i in range(365)) if d.weekday() >= 5]
    idx = sorted(rng.choice(len(weekend), 6, replace=False))
    return _gl(
        rng, 6, "night_postings",
        posting_date=[weekend[i].isoformat() for i in idx], posting_time=times(rng, 6, 1.5, 3.5),
        created_by="U105", dr_account="2300", cr_account="1000",
        amount=money(rng.uniform(2_100, 4_600, 6)), description="Payroll correction",
    )  # fmt: skip


def expense_capitalisation(rng: np.random.Generator) -> pd.DataFrame:
    dates = ["2025-12-29", "2025-12-30", "2025-12-30", "2025-12-31"]
    return _gl(
        rng, 4, "expense_capitalisation",
        posting_date=dates, posting_time=times(rng, 4, 17.0, 19.5), created_by="U107", approved_by="U108",
        dr_account="1500", cr_account=["6200", "6400", "6200", "6400"],
        amount=[38_500.0, 52_300.0, 61_750.0, 79_900.0], description="Reclass to fixed assets",
    )  # fmt: skip


def duplicate_invoices(rng: np.random.Generator, entries: pd.DataFrame) -> pd.DataFrame:
    pool = entries[(entries["entry_type"] == "purchase_invoice") & (entries["created_by"] == "U102")
                   & (entries["posting_date"] < f"{YEAR}-12-01") & (entries["amount"] > 800)]  # fmt: skip
    picks = pool.sample(6, random_state=int(rng.integers(0, 10_000)))
    out = picks.copy()
    out["posting_date"] = [
        next_business_day(date.fromisoformat(d) + timedelta(days=int(rng.integers(2, 9)))).isoformat()
        for d in picks["posting_date"]
    ]
    out["posting_time"] = times(rng, 6, 9.0, 17.0)
    # Three exact copies, three with a slightly changed reference to get past a simple duplicate check.
    out["reference"] = [ref if i < 3 else f"{ref}-1" for i, ref in enumerate(out["reference"])]
    out["description"] = [
        f"Invoice {r} {d.split(' ', 2)[-1]}" for r, d in zip(out["reference"], picks["description"], strict=True)
    ]
    out["_scheme"] = "duplicate_invoices"
    return out


def misc_cash(rng: np.random.Generator) -> pd.DataFrame:
    days = sorted(rng.choice(len(BUSINESS_DAYS), 6, replace=False))
    return _gl(
        rng, 6, "misc_cash",
        posting_date=[BUSINESS_DAYS[i].isoformat() for i in days], posting_time=times(rng, 6, 10.0, 16.0),
        created_by="U103", dr_account="7000", cr_account="1000", amount=money(rng.uniform(1_200, 4_800, 6)),
        description=["adj", "corr", "misc", "see email", "various", "adj 2"],
    )  # fmt: skip


def inject_schemes(
    entries: pd.DataFrame, vendors: pd.DataFrame, customers: pd.DataFrame, rng: np.random.Generator, limit: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ghost_entries, vendors = ghost_vendor(rng, vendors)
    planted = [
        fictitious_revenue(rng, customers),
        split_payments(rng, vendors, limit),
        ghost_entries,
        self_approval(rng),
        night_postings(rng),
        expense_capitalisation(rng),
        duplicate_invoices(rng, entries),
        misc_cash(rng),
    ]
    return pd.concat([entries, *planted], ignore_index=True), vendors
