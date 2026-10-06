"""Generate a year of bookkeeping for a fictional Dutch logistics company.

The ledger is built from normal business processes (sales, receipts, purchases,
payments, payroll, depreciation, month-end journals). It deliberately contains
legitimate entries that look odd to simple audit rules: nightly system batches,
Saturday invoicing, round rent payments, month-end accruals and their reversals.
Fraud schemes are added afterwards by ``ledgerlens.data.schemes``.

All names, amounts and identifiers are fictional.
"""

from __future__ import annotations

import logging
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ledgerlens.data import chart
from ledgerlens.data.common import (
    BUSINESS_DAYS,
    HOLIDAYS,
    YEAR,
    all_days,
    is_business_day,
    last_business_day,
    money,
    next_business_day,
    times,
)
from ledgerlens.data.common import frame as _frame
from ledgerlens.data.schemes import inject_schemes

logger = logging.getLogger(__name__)

USERS = [
    ("SYS-SALES", "Sales interface", "system"),
    ("SYS-BANK", "Bank interface", "system"),
    ("SYS-PAY", "Payroll interface", "system"),
    ("SYS-ASSET", "Fixed asset module", "system"),
    ("U101", "Anna Visser", "Accounts payable clerk"),
    ("U102", "Bram de Groot", "Accounts payable clerk"),
    ("U103", "Chloe Mulder", "General ledger accountant"),
    ("U104", "Dennis Smit", "General ledger accountant"),
    ("U105", "Eva Bos", "Payroll officer"),
    ("U106", "Floor Meijer", "Finance manager"),
    ("U107", "Gerrit Dekker", "Financial controller"),
    ("U108", "Hanna Vos", "Chief financial officer"),
]
APPROVERS = {"U106", "U107", "U108"}

_SURNAMES = [
    "Bakker",
    "Jansen",
    "Visser",
    "Smit",
    "Meijer",
    "Mulder",
    "Bos",
    "Vos",
    "Peters",
    "Hendriks",
    "Dekker",
    "Brouwer",
    "Kok",
    "Jacobs",
    "Vermeulen",
    "Willems",
    "Hoekstra",
    "Koster",
    "Prins",
    "Huisman",
    "Peeters",
    "Kuiper",
    "Veenstra",
    "Post",
    "Wouters",
    "Schouten",
    "Dijkstra",
    "Kramer",
    "Maas",
    "Verhoeven",
    "Bosman",
    "Kuipers",
    "Groen",
    "Lammers",
    "Blom",
    "Wolters",
]


@dataclass
class Ledger:
    entries: pd.DataFrame
    vendors: pd.DataFrame
    customers: pd.DataFrame
    users: pd.DataFrame
    ground_truth: pd.DataFrame


# --------------------------------------------------------------------------- master data
def make_customers(rng: np.random.Generator, n: int = 350) -> pd.DataFrame:
    suffixes = ["Transport", "Retail", "Foods", "Bouw", "Techniek", "Groothandel", "Logistiek"]
    names = [f"{rng.choice(_SURNAMES)} {rng.choice(suffixes)} B.V." for _ in range(n)]
    weights = rng.pareto(1.3, n) + 0.05
    return pd.DataFrame(
        {"customer_id": [f"C{i:04d}" for i in range(1, n + 1)], "name": names, "weight": weights / weights.sum()}
    )


def make_vendors(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    vid = 1

    def add(name: str, category: str, pattern: str, mu: float = 0.0, sigma: float = 0.0,
            fixed: float = 0.0, vat_rate: float = 0.21, weight: float = 1.0,
            created: date | None = None, created_by: str | None = None) -> None:  # fmt: skip
        nonlocal vid
        if created is None:
            created = date(int(rng.integers(2016, 2025)), int(rng.integers(1, 13)), int(rng.integers(1, 28)))
        rows.append(
            {
                "vendor_id": f"V{vid:04d}",
                "name": name,
                "category": category,
                "account": chart.VENDOR_CATEGORY_ACCOUNT[category],
                "pattern": pattern,
                "mu": mu,
                "sigma": sigma,
                "fixed_amount": fixed,
                "vat_rate": vat_rate,
                "weight": weight,
                "created_date": created.isoformat(),
                "created_by": created_by or str(rng.choice(["U106", "U106", "U107"])),
            }
        )
        vid += 1

    def name(suffix: str) -> str:
        return f"{rng.choice(_SURNAMES)} {suffix}"

    for _ in range(40):
        add(name("Transport B.V."), "haulage subcontractor", "variable", 7.6, 0.8, weight=float(rng.uniform(0.5, 3)))
    for _ in range(8):
        add(name("Brandstoffen"), "fuel", "variable", 7.1, 0.6, weight=float(rng.uniform(1, 3)))
    for _ in range(12):
        add(name("Truckservice"), "vehicle maintenance", "variable", 6.6, 0.9, weight=float(rng.uniform(0.5, 2)))
    for _ in range(12):
        add(name("IT Services"), "it services", "variable", 7.0, 1.0, weight=float(rng.uniform(0.3, 1)))
    for _ in range(6):
        add(name("Software"), "software", "monthly", fixed=float(rng.choice([750, 1000, 1500, 2000])))
    for _ in range(6):
        add(name("Consultancy"), "consulting", "variable", 8.2, 0.6, weight=float(rng.uniform(0.1, 0.4)))
    for _ in range(10):
        add(name("Media"), "marketing", "variable", 6.9, 0.9, weight=float(rng.uniform(0.2, 0.8)))
    for _ in range(8):
        add(name("Reizen"), "travel", "variable", 5.8, 0.8, weight=float(rng.uniform(0.3, 1)))
    for _ in range(12):
        add(name("Kantoorartikelen"), "office supplies", "variable", 5.2, 0.8, weight=float(rng.uniform(0.3, 1)))
    for _ in range(2):
        add(name("Schoonmaak"), "cleaning", "weekly", fixed=float(rng.choice([450, 520])))
    add("Havenvast Vastgoed B.V.", "rent", "monthly", fixed=12_500.0, vat_rate=0.0)
    for _ in range(3):
        add(name("Lease"), "vehicle lease", "monthly", fixed=float(rng.choice([1500, 2250, 3000])))
    add(name("Verzekeringen"), "insurance", "monthly", fixed=4_150.0, vat_rate=0.0)
    # Legitimate new suppliers during the year, created by the finance manager.
    for _ in range(10):
        created = date(YEAR, int(rng.integers(2, 11)), int(rng.integers(1, 28)))
        add(name("Transport B.V."), "haulage subcontractor", "variable", 7.4, 0.7,
            weight=float(rng.uniform(0.3, 1.0)), created=created, created_by="U106")  # fmt: skip
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- business processes
def sales_and_receipts(rng: np.random.Generator, customers: pd.DataFrame, scale: float) -> list[pd.DataFrame]:
    days = [d for d in all_days() if (is_business_day(d) or d.weekday() == 5) and d not in HOLIDAYS]
    per_day = np.array([0.35 if d.weekday() == 5 else 1.0 for d in days])
    n = int(70_000 * scale)
    day_idx = rng.choice(len(days), n, p=per_day / per_day.sum())
    inv_dates = [days[i] for i in day_idx]
    net = money(np.exp(rng.normal(7.4, 0.9, n)))
    vat = money(net * 0.21)
    gross = money(net + vat)
    cust = rng.choice(customers["customer_id"], n, p=customers["weight"])
    refs = [f"INV-25{i:06d}" for i in range(1, n + 1)]
    sales = _frame(
        posting_date=[d.isoformat() for d in inv_dates],
        posting_time=times(rng, n, 21.0, 23.5),
        source="SALES",
        entry_type="sales_invoice",
        created_by="SYS-SALES",
        approved_by="",
        dr_account="1100",
        cr_account="4000",
        amount=gross,
        vat_amount=vat,
        vat_side="cr",
        customer_id=cust,
        reference=refs,
        description=[f"Sales invoice {r}" for r in refs],
    )
    paid = rng.random(n) < 0.93
    delay = rng.integers(10, 60, n)
    r_dates, keep = [], []
    for i in range(n):
        if not paid[i]:
            continue
        d = next_business_day(inv_dates[i] + timedelta(days=int(delay[i])))
        if d.year == YEAR:
            r_dates.append(d)
            keep.append(i)
    keep_arr = np.array(keep)
    receipts = _frame(
        posting_date=[d.isoformat() for d in r_dates],
        posting_time=times(rng, len(keep), 6.0, 7.5),
        source="BANK",
        entry_type="customer_receipt",
        created_by="SYS-BANK",
        approved_by="",
        dr_account="1000",
        cr_account="1100",
        amount=gross[keep_arr],
        customer_id=cust[keep_arr],
        reference=[refs[i] for i in keep],
        description=[f"Receipt {refs[i]}" for i in keep],
    )
    # A few manual credit notes by the general ledger team (legitimate manual revenue entries).
    m = max(3, int(40 * scale))
    idx = rng.choice(n, m, replace=False)
    cn_dates = [next_business_day(inv_dates[i] + timedelta(days=int(rng.integers(3, 20)))) for i in idx]
    cn_net = money(net[idx] * rng.uniform(0.1, 1.0, m))
    cn_vat = money(cn_net * 0.21)
    credit_notes = _frame(
        posting_date=[d.isoformat() if d.year == YEAR else f"{YEAR}-12-30" for d in cn_dates],
        posting_time=times(rng, m, 9.0, 17.0),
        source="GL",
        entry_type="credit_note",
        created_by=rng.choice(["U103", "U104"], m),
        approved_by="",
        dr_account="4000",
        cr_account="1100",
        amount=money(cn_net + cn_vat),
        vat_amount=cn_vat,
        vat_side="dr",
        customer_id=cust[idx],
        reference=[f"CN-{refs[i]}" for i in idx],
        description=[f"Credit note on {refs[i]}" for i in idx],
    )
    return [sales, receipts, credit_notes]


def purchases_and_payments(
    rng: np.random.Generator, vendors: pd.DataFrame, scale: float, limit: float
) -> list[pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    variable = vendors[vendors["pattern"] == "variable"].reset_index(drop=True)
    n_var = int(35_000 * scale)
    w = variable["weight"].to_numpy()
    vidx = rng.choice(len(variable), n_var, p=w / w.sum())
    start_idx = np.array(
        [bisect_left(BUSINESS_DAYS, max(date.fromisoformat(c), date(YEAR, 1, 1))) for c in variable["created_date"]]
    )
    span = len(BUSINESS_DAYS) - start_idx[vidx]
    day_idx = start_idx[vidx] + (rng.random(n_var) * span).astype(int)
    nets = np.exp(rng.normal(variable["mu"].to_numpy()[vidx], variable["sigma"].to_numpy()[vidx]))
    records = variable.to_dict("records")
    rows = [
        {"vendor": records[v], "date": BUSINESS_DAYS[d], "net": float(x)}
        for v, d, x in zip(vidx, day_idx, nets, strict=True)
    ]
    for v in vendors[vendors["pattern"].isin(["monthly", "weekly"])].to_dict("records"):
        if v["pattern"] == "monthly":
            dates = [next_business_day(date(YEAR, m, 1)) for m in range(1, 13)]
        else:
            dates = [d for d in BUSINESS_DAYS if d.weekday() == 0]
        rows += [{"vendor": v, "date": d, "net": float(v["fixed_amount"])} for d in dates]

    k = len(rows)
    net = money([r["net"] for r in rows])
    vat_rate = np.array([float(r["vendor"]["vat_rate"]) for r in rows])
    vat = money(net * vat_rate)
    gross = money(net + vat)
    clerk = rng.choice(["U101", "U102"], k)
    vend_ids = [r["vendor"]["vendor_id"] for r in rows]
    refs = [f"{r['vendor']['vendor_id']}-{rng.integers(10000, 99999)}" for r in rows]
    approver = np.where(gross >= limit, rng.choice(["U106", "U107"], k), "")
    inv = _frame(
        posting_date=[r["date"].isoformat() for r in rows],
        posting_time=times(rng, k, 8.5, 17.5),
        source="AP",
        entry_type="purchase_invoice",
        created_by=clerk,
        approved_by=approver,
        dr_account=[r["vendor"]["account"] for r in rows],
        cr_account="2000",
        amount=gross,
        vat_amount=vat,
        vat_side="dr",
        vendor_id=vend_ids,
        reference=refs,
        description=[f"Invoice {ref} {r['vendor']['name']}" for ref, r in zip(refs, rows, strict=True)],
    )

    # Payments: payment runs on Tuesday and Thursday, about 30 days after the invoice.
    pay_rows = []
    manual = rng.random(k) < 0.01
    for i, r in enumerate(rows):
        due = r["date"] + timedelta(days=int(rng.integers(20, 40)))
        d = next_business_day(due)
        if not manual[i]:
            while d.weekday() not in (1, 3) or not is_business_day(d):
                d += timedelta(days=1)
        if d.year != YEAR:
            continue
        pay_rows.append((i, d, bool(manual[i])))
    idx = np.array([p[0] for p in pay_rows])
    is_manual = np.array([p[2] for p in pay_rows])
    payments = _frame(
        posting_date=[p[1].isoformat() for p in pay_rows],
        posting_time=[
            t_man if m else t_run
            for t_man, t_run, m in zip(
                times(rng, len(pay_rows), 9.0, 17.0), times(rng, len(pay_rows), 10.0, 11.5), is_manual, strict=True
            )
        ],
        source=np.where(is_manual, "BANK-MANUAL", "PAYRUN"),
        entry_type=np.where(is_manual, "manual_payment", "vendor_payment"),
        created_by=rng.choice(["U101", "U102"], len(pay_rows)),
        approved_by=np.where(gross[idx] >= limit, "U106", ""),
        dr_account="2000",
        cr_account="1000",
        amount=gross[idx],
        vendor_id=[vend_ids[i] for i in idx],
        reference=[refs[i] for i in idx],
        description=[f"Payment {refs[i]}" for i in idx],
    )
    return [inv, payments]


def payroll_and_assets(rng: np.random.Generator) -> list[pd.DataFrame]:
    frames = []
    for m in range(1, 13):
        pay_day = (
            next_business_day(date(YEAR, m, 25)) if date(YEAR, m, 25).weekday() < 5 else last_business_day(YEAR, m)
        )
        gross = float(rng.uniform(185_000, 196_000))
        frames.append(_frame(posting_date=[pay_day.isoformat()], posting_time=times(rng, 1, 1.0, 2.0), source="PAYROLL",
                             entry_type="payroll", created_by="SYS-PAY", dr_account="6000", cr_account="2300",
                             amount=money([gross]), description=[f"Salaries {YEAR}-{m:02d}"]))  # fmt: skip
        frames.append(_frame(posting_date=[pay_day.isoformat()], posting_time=times(rng, 1, 6.0, 7.0), source="BANK",
                             entry_type="salary_payment", created_by="SYS-BANK", dr_account="2300", cr_account="1000",
                             amount=money([gross * 0.68]), description=[f"Net salaries {YEAR}-{m:02d}"]))  # fmt: skip
        frames.append(_frame(posting_date=[(pay_day - timedelta(days=1)).isoformat()], posting_time=times(rng, 1, 10, 15),
                             source="GL", entry_type="manual_journal", created_by="U105", approved_by="U106",
                             dr_account="6000", cr_account="2300", amount=money([rng.uniform(20_000, 24_000)]),
                             description=[f"Pension and social charges {YEAR}-{m:02d}"]))  # fmt: skip
        tax_day = next_business_day(date(YEAR, m, 28) if m != 2 else date(YEAR, 2, 27))
        frames.append(_frame(posting_date=[tax_day.isoformat()], posting_time=times(rng, 1, 9, 12), source="GL",
                             entry_type="manual_journal", created_by="U105", approved_by="U106", dr_account="2300",
                             cr_account="1000", amount=money([gross * 0.32]),
                             description=[f"Payroll tax payment {YEAR}-{m:02d}"]))  # fmt: skip
        month_end = last_business_day(YEAR, m)
        for asset, base in (
            ("vehicles", 38_420.15),
            ("buildings fit-out", 6_210.40),
            ("IT hardware", 2_955.80),
            ("warehouse equipment", 4_870.25),
        ):
            frames.append(_frame(posting_date=[month_end.isoformat()], posting_time=times(rng, 1, 23.0, 23.6), source="ASSET",
                                 entry_type="depreciation", created_by="SYS-ASSET", dr_account="6800", cr_account="1600",
                                 amount=[base], description=[f"Depreciation {asset} {YEAR}-{m:02d}"]))  # fmt: skip
        first = next_business_day(date(YEAR, m, 1))
        frames.append(_frame(posting_date=[first.isoformat()], posting_time=times(rng, 1, 6.0, 7.0), source="BANK",
                             entry_type="bank_charges", created_by="SYS-BANK", dr_account="6900", cr_account="1000",
                             amount=money([rng.uniform(150, 420)]), description=[f"Bank charges {YEAR}-{m:02d}"]))  # fmt: skip
    return frames


def manual_journals(rng: np.random.Generator, scale: float, limit: float) -> list[pd.DataFrame]:
    """Month-end accruals and reversals, reclasses, prepayments and year-end close entries."""
    rows: list[dict[str, Any]] = []
    accrual_accounts = ["6100", "6200", "6300", "6400", "6500", "6700", "6900"]

    def approver_for(amount: float, creator: str) -> str:
        return str(rng.choice(["U107", "U106"])) if amount >= limit else ""

    for m in range(1, 13):
        month_end = last_business_day(YEAR, m)
        for _ in range(max(2, int(16 * scale))):
            acct = str(rng.choice(accrual_accounts))
            amount = float(np.exp(rng.normal(8.2, 0.9)))
            if rng.random() < 0.6:
                amount = round(amount / 500) * 500 or 500.0
            creator = str(rng.choice(["U103", "U104"]))
            d = month_end - timedelta(days=int(rng.integers(0, 3)))
            d = d if is_business_day(d) else month_end
            hour = (9.0, 19.0) if rng.random() < 0.8 else (19.0, 21.5)
            label = chart.ACCOUNTS[acct]
            rows.append({"date": d, "time": times(rng, 1, *hour)[0], "creator": creator, "dr": acct, "cr": "2100",
                         "amount": amount, "approver": approver_for(amount, creator), "desc": f"Accrual {label.lower()} {YEAR}-{m:02d}"})  # fmt: skip
            if m < 12:
                rd = next_business_day(date(YEAR, m + 1, 1) + timedelta(days=int(rng.integers(0, 2))))
                rows.append({"date": rd, "time": times(rng, 1, 9, 17)[0], "creator": creator, "dr": "2100", "cr": acct,
                             "amount": amount, "approver": approver_for(amount, creator), "desc": f"Reversal accrual {label.lower()} {YEAR}-{m:02d}"})  # fmt: skip
    for _ in range(int(600 * scale)):
        a, b = rng.choice(accrual_accounts, 2, replace=False)
        amount = float(np.exp(rng.normal(7.3, 1.0)))
        d = BUSINESS_DAYS[int(rng.integers(0, len(BUSINESS_DAYS)))]
        rows.append({"date": d, "time": times(rng, 1, 9, 17.5)[0], "creator": str(rng.choice(["U103", "U104"])), "dr": str(a),
                     "cr": str(b), "amount": amount, "approver": approver_for(amount, ""), "desc": f"Reclass {chart.ACCOUNTS[str(b)].lower()} to {chart.ACCOUNTS[str(a)].lower()}"})  # fmt: skip
    for _ in range(int(100 * scale)):
        d = BUSINESS_DAYS[int(rng.integers(0, len(BUSINESS_DAYS)))]
        amount = float(np.exp(rng.normal(7.8, 0.7)))
        rows.append({"date": d, "time": times(rng, 1, 9, 17)[0], "creator": str(rng.choice(["U103", "U104"])), "dr": "6300",
                     "cr": "1300", "amount": amount, "approver": approver_for(amount, ""), "desc": "Release prepaid software licences"})  # fmt: skip
    for m in range(1, 13):
        amount = float(rng.uniform(4_000, 15_000))
        rows.append({"date": last_business_day(YEAR, m), "time": times(rng, 1, 14, 17)[0], "creator": "U103", "dr": "5000",
                     "cr": "1200", "amount": amount, "approver": approver_for(amount, ""), "desc": f"Inventory count adjustment {YEAR}-{m:02d}"})  # fmt: skip
    for _ in range(max(3, int(60 * scale))):
        a, b = rng.choice(accrual_accounts, 2, replace=False)
        d = BUSINESS_DAYS[int(rng.integers(0, len(BUSINESS_DAYS)))]
        rows.append({"date": d, "time": times(rng, 1, 9, 17)[0], "creator": str(rng.choice(["U103", "U104"])), "dr": str(a),
                     "cr": str(b), "amount": float(np.exp(rng.normal(6.0, 0.8))), "approver": "", "desc": str(rng.choice(["Correction", "Correction posting", "Adj. coding error"]))})  # fmt: skip
    # Year-end close: weekend work on 27 and 28 December, plus accrued revenue booked by the controller.
    for _ in range(max(4, int(15 * scale))):
        a = str(rng.choice(accrual_accounts))
        d = date(YEAR, 12, int(rng.choice([27, 28])))
        amount = float(round(np.exp(rng.normal(8.5, 0.6)) / 500) * 500 or 500)
        rows.append({"date": d, "time": times(rng, 1, 10, 15)[0], "creator": str(rng.choice(["U103", "U104"])), "dr": a,
                     "cr": "2100", "amount": amount, "approver": approver_for(amount, ""), "desc": f"Year-end accrual {chart.ACCOUNTS[a].lower()}"})  # fmt: skip
    rows.append({"date": date(YEAR, 12, 31), "time": "16:42:10", "creator": "U107", "dr": "1100", "cr": "4000",
                 "amount": 85_000.0, "approver": "U108", "desc": "Accrued revenue December, signed delivery notes"})  # fmt: skip
    # A control weakness, not fraud: the controller approves some of his own accruals.
    for m in rng.choice(range(1, 13), 5, replace=False):
        amount = float(rng.uniform(10_500, 18_000))
        rows.append({"date": last_business_day(YEAR, int(m)), "time": times(rng, 1, 10, 17)[0], "creator": "U107", "dr": "6400",
                     "cr": "2100", "amount": amount, "approver": "U107", "desc": f"Accrual audit and advisory fees {YEAR}-{int(m):02d}"})  # fmt: skip

    return [
        _frame(
            posting_date=[r["date"].isoformat() for r in rows],
            posting_time=[r["time"] for r in rows],
            source="GL",
            entry_type="manual_journal",
            created_by=[r["creator"] for r in rows],
            approved_by=[r["approver"] for r in rows],
            dr_account=[r["dr"] for r in rows],
            cr_account=[r["cr"] for r in rows],
            amount=money([r["amount"] for r in rows]),
            description=[r["desc"] for r in rows],
        )
    ]


# --------------------------------------------------------------------------- assemble
def generate_ledger(seed: int = 7, scale: float = 1.0, limit: float = 10_000.0) -> Ledger:
    rng = np.random.default_rng(seed)
    customers = make_customers(rng)
    vendors = make_vendors(rng)
    frames = []
    frames += sales_and_receipts(rng, customers, scale)
    frames += purchases_and_payments(rng, vendors, scale, limit)
    frames += payroll_and_assets(rng)
    frames += manual_journals(rng, scale, limit)
    entries = pd.concat(frames, ignore_index=True)
    entries["_scheme"] = ""

    entries, vendors = inject_schemes(entries, vendors, customers, rng, limit)

    entries = entries.sort_values(["posting_date", "posting_time"], kind="stable").reset_index(drop=True)
    entries.insert(0, "entry_id", [f"JE{i:06d}" for i in range(1, len(entries) + 1)])
    entries["period"] = entries["posting_date"].str.slice(5, 7).astype(int)
    truth = entries.loc[entries["_scheme"] != "", ["entry_id", "_scheme"]].rename(columns={"_scheme": "scheme"})
    entries = entries.drop(columns="_scheme")
    users = pd.DataFrame(USERS, columns=["user_id", "name", "role"])
    users["can_approve"] = users["user_id"].isin(APPROVERS)
    logger.info("Generated %d journal entries with %d planted fraud entries", len(entries), len(truth))
    return Ledger(
        entries,
        vendors.drop(columns=["mu", "sigma", "weight"]),
        customers.drop(columns="weight"),
        users,
        truth.reset_index(drop=True),
    )


def save_ledger(ledger: Ledger, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    ledger.entries.to_csv(data_dir / "journal_entries.csv", index=False)
    ledger.vendors.to_csv(data_dir / "vendors.csv", index=False)
    ledger.customers.to_csv(data_dir / "customers.csv", index=False)
    ledger.users.to_csv(data_dir / "users.csv", index=False)
    # Kept apart on purpose: nothing in the detection pipeline reads this file.
    ledger.ground_truth.to_csv(data_dir / "ground_truth.csv", index=False)


def load_ledger(data_dir: Path) -> Ledger:
    if not (data_dir / "journal_entries.csv").is_file():
        raise FileNotFoundError(f"No ledger in {data_dir}. Run: ledgerlens generate")
    str_cols = {c: str for c in ("approved_by", "vendor_id", "customer_id", "reference", "dr_account", "cr_account")}
    entries = pd.read_csv(data_dir / "journal_entries.csv", dtype=str_cols, keep_default_na=False)
    vendors = pd.read_csv(data_dir / "vendors.csv", dtype={"account": str}, keep_default_na=False)
    customers = pd.read_csv(data_dir / "customers.csv", keep_default_na=False)
    users = pd.read_csv(data_dir / "users.csv", keep_default_na=False)
    truth_path = data_dir / "ground_truth.csv"
    truth = pd.read_csv(truth_path) if truth_path.is_file() else pd.DataFrame(columns=["entry_id", "scheme"])
    return Ledger(entries, vendors, customers, users, truth)
