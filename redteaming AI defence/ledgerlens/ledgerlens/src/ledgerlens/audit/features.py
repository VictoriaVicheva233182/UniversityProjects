"""Turn journal entries into risk features.

Every feature is a standard journal entry risk factor (timing, amount, user,
approval, supplier, account combination). The important difference with plain
rules: many features are measured against what is normal *for this user* or
*for this account combination*, so a nightly system batch is not unusual, but
a payroll officer posting at 2 a.m. is.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ledgerlens.data.common import HOLIDAYS

VAGUE_RE = re.compile(r"^(adj|corr|misc|test|see email|various|n/?a|x+)(\.| \d+)?$", re.IGNORECASE)
MANUAL_SOURCES = {"GL", "BANK-MANUAL"}

# Features the anomaly models see. Context columns (counts, medians) are kept for explanations.
MODEL_FEATURES = [
    "log_amount", "round_1000", "round_100", "below_limit", "off_hours", "user_hour_dev",
    "weekend_surprise", "is_manual", "quarter_end", "pair_rarity", "user_pair_rarity",
    "amount_z_in_pair", "self_approved", "missing_approval", "poster_created_vendor",
    "new_vendor", "dup_surprise", "dup_reference", "same_day_vendor_payments", "reversed", "vague_description",
    "manual_revenue", "manual_cash_out",
]  # fmt: skip


def _robust_z(values: pd.Series, groups: pd.Series) -> pd.Series:
    med = values.groupby(groups).transform("median")
    mad = (values - med).abs().groupby(groups).transform("median") * 1.4826
    return ((values - med) / (mad + 0.25)).clip(-10, 10)


def build_features(entries: pd.DataFrame, vendors: pd.DataFrame, limit: float = 10_000.0) -> pd.DataFrame:
    e = entries
    f = pd.DataFrame({"entry_id": e["entry_id"].to_numpy()})
    dt = pd.to_datetime(e["posting_date"] + " " + e["posting_time"])
    date = dt.dt.normalize()
    hour = dt.dt.hour + dt.dt.minute / 60
    amount = e["amount"].astype(float)
    user = e["created_by"].astype(str)
    pair = e["dr_account"].astype(str) + "/" + e["cr_account"].astype(str)
    is_system = user.str.startswith("SYS")
    is_manual = e["source"].isin(MANUAL_SOURCES)

    f["hour"] = hour.round(2).to_numpy()
    f["weekday"] = dt.dt.dayofweek.to_numpy()
    holidays = pd.to_datetime(sorted(HOLIDAYS))
    weekend = (dt.dt.dayofweek >= 5) | date.isin(holidays)
    f["weekend_or_holiday"] = weekend.astype(int).to_numpy()
    f["off_hours"] = ((hour < 7) | (hour >= 20)).astype(int).to_numpy()
    f["log_amount"] = np.log10(amount.clip(lower=1)).to_numpy()
    f["round_1000"] = ((amount % 1000 == 0) & (amount >= 1000)).astype(int).to_numpy()
    f["round_100"] = ((amount % 100 == 0) & (amount >= 100)).astype(int).to_numpy()
    f["below_limit"] = ((amount >= 0.9 * limit) & (amount < limit) & ~is_system).astype(int).to_numpy()
    f["is_manual"] = is_manual.astype(int).to_numpy()
    f["is_system"] = is_system.astype(int).to_numpy()

    month_end = date + pd.offsets.MonthEnd(0)
    f["days_to_month_end"] = (month_end - date).dt.days.to_numpy()
    f["quarter_end"] = (dt.dt.month.isin([3, 6, 9, 12]) & ((month_end - date).dt.days <= 1)).astype(int).to_numpy()

    # User behaviour: when does this person normally post?
    user_median = hour.groupby(user).transform("median")
    q75 = hour.groupby(user).transform(lambda s: s.quantile(0.75))
    q25 = hour.groupby(user).transform(lambda s: s.quantile(0.25))
    f["user_median_hour"] = user_median.round(2).to_numpy()
    f["user_hour_dev"] = ((hour - user_median).abs() / (q75 - q25).clip(lower=1.0)).clip(0, 12).to_numpy()
    weekend_share = weekend.groupby(user).transform("mean")
    f["weekend_surprise"] = (weekend.astype(float) * (1 - weekend_share)).to_numpy()

    # Account combinations, overall and per user.
    pair_count = pair.map(pair.value_counts())
    user_count = user.map(user.value_counts())
    user_pair_count = (user + "|" + pair).map((user + "|" + pair).value_counts())
    f["pair"] = pair.to_numpy()
    f["pair_count"] = pair_count.to_numpy()
    f["user_entries"] = user_count.to_numpy()
    f["user_pair_count"] = user_pair_count.to_numpy()
    f["pair_rarity"] = (-np.log10(pair_count / len(e))).to_numpy()
    f["user_pair_rarity"] = (-np.log10(user_pair_count / user_count)).to_numpy()
    f["amount_z_in_pair"] = _robust_z(np.log10(amount.clip(lower=1)), pair).to_numpy()

    approver = e["approved_by"].astype(str)
    f["self_approved"] = ((approver == user) & (approver != "")).astype(int).to_numpy()
    f["missing_approval"] = ((amount >= limit) & (approver == "") & ~is_system).astype(int).to_numpy()

    # Supplier context.
    vend = vendors.set_index("vendor_id")
    vid = e["vendor_id"].astype(str)
    has_vendor = vid != ""
    created = pd.to_datetime(vid.map(vend["created_date"]), errors="coerce")
    age = (date - created).dt.days
    f["vendor_age_days"] = age.fillna(-1).astype(int).to_numpy()
    f["new_vendor"] = (has_vendor & (age < 60)).astype(int).to_numpy()
    f["poster_created_vendor"] = (has_vendor & (vid.map(vend["created_by"]) == user)).astype(int).to_numpy()

    # Same supplier, same amount, within 14 days (purchase invoices only).
    inv = e["entry_type"] == "purchase_invoice"
    dup = np.zeros(len(e), dtype=int)
    inv_df = pd.DataFrame(
        {"idx": np.where(inv)[0], "key": (vid + "|" + amount.astype(str))[inv].to_numpy(), "d": date[inv].to_numpy()}
    )
    for _, g in inv_df.groupby("key"):
        if len(g) < 2:
            continue
        days = g["d"].to_numpy().astype("datetime64[D]").astype(int)
        for i, d0 in zip(g["idx"].to_numpy(), days, strict=True):
            dup[i] = int((np.abs(days - d0) <= 14).sum() - 1)
    f["dup_vendor_amount"] = dup
    ref_key = vid + "|" + e["reference"].astype(str)
    f["dup_reference"] = np.where(inv, ref_key.map(ref_key[inv].value_counts()).fillna(1) - 1, 0).astype(int)
    # Weekly cleaning invoices repeat the same amount all the time; a haulage firm does not.
    repeat_share = pd.Series(dup > 0, index=e.index).where(inv).groupby(vid).transform("mean").fillna(0)
    f["vendor_repeat_share"] = repeat_share.round(3).to_numpy()
    f["dup_surprise"] = ((dup > 0) * (1 - repeat_share)).to_numpy()

    # Several payments by one person to one supplier on one day.
    pay = e["entry_type"].isin(["manual_payment", "vendor_payment"]) & has_vendor
    key = user + "|" + vid + "|" + e["posting_date"]
    counts = key[pay].map(key[pay].value_counts())
    f["same_day_vendor_payments"] = (counts.reindex(e.index).fillna(1) - 1).astype(int).to_numpy()

    # Reversed: an entry with swapped accounts and the same amount within 10 days.
    swap = pd.DataFrame({"i": np.arange(len(e)), "k": pair.to_numpy(), "a": amount.to_numpy(), "d": date.to_numpy()})
    rev = swap.assign(k=(e["cr_account"].astype(str) + "/" + e["dr_account"].astype(str)).to_numpy())
    merged = swap.merge(rev, on=["k", "a"], suffixes=("", "_r"))
    close = (merged["d"] - merged["d_r"]).abs().dt.days <= 10
    reversed_idx = set(merged.loc[close & (merged["i"] != merged["i_r"]), "i"])
    f["reversed"] = np.isin(np.arange(len(e)), list(reversed_idx)).astype(int)

    desc = e["description"].astype(str).str.strip()
    f["vague_description"] = (desc.str.match(VAGUE_RE) | (desc.str.len() < 5)).astype(int).to_numpy()
    revenue = (e["dr_account"] == "4000") | (e["cr_account"] == "4000")
    f["manual_revenue"] = (is_manual & revenue).astype(int).to_numpy()
    f["manual_cash_out"] = ((e["source"] == "GL") & (e["cr_account"] == "1000")).astype(int).to_numpy()
    return f
