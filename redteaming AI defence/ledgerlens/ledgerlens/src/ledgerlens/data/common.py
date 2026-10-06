"""Calendar and frame helpers shared by the generator and the fraud schemes."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

YEAR = 2025
HOLIDAYS = {
    date(2025, 1, 1), date(2025, 4, 21), date(2025, 4, 26), date(2025, 5, 5),
    date(2025, 5, 29), date(2025, 6, 9), date(2025, 12, 25), date(2025, 12, 26),
}  # fmt: skip

ENTRY_COLUMNS = [
    "posting_date", "posting_time", "source", "entry_type", "created_by", "approved_by",
    "dr_account", "cr_account", "amount", "vat_amount", "vat_side", "vendor_id",
    "customer_id", "reference", "description", "_scheme",
]  # fmt: skip


def all_days() -> list[date]:
    d, out = date(YEAR, 1, 1), []
    while d.year == YEAR:
        out.append(d)
        d += timedelta(days=1)
    return out


def is_business_day(d: date) -> bool:
    return d.weekday() < 5 and d not in HOLIDAYS


BUSINESS_DAYS = [d for d in all_days() if is_business_day(d)]


def next_business_day(d: date) -> date:
    while not is_business_day(d):
        d += timedelta(days=1)
    return d


def last_business_day(year: int, month: int) -> date:
    d = (date(year, month + 1, 1) if month < 12 else date(year + 1, 1, 1)) - timedelta(days=1)
    while not is_business_day(d):
        d -= timedelta(days=1)
    return d


def times(rng: np.random.Generator, n: int, start_h: float, end_h: float) -> list[str]:
    secs = rng.uniform(start_h * 3600, end_h * 3600, n).astype(int) % 86400
    return [f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}" for s in secs]


def money(x: object) -> np.ndarray:
    return np.round(np.asarray(x, dtype=float), 2)


def frame(**cols: object) -> pd.DataFrame:
    """Build an entries frame with every column present, in a fixed order."""
    n = max((len(v) for v in cols.values() if isinstance(v, (list, np.ndarray, pd.Series))), default=1)
    df = pd.DataFrame({k: (v if isinstance(v, (list, np.ndarray, pd.Series)) else [v] * n) for k, v in cols.items()})
    for col in ENTRY_COLUMNS:
        if col not in df:
            df[col] = 0.0 if col == "vat_amount" else ""
    return df[ENTRY_COLUMNS]
