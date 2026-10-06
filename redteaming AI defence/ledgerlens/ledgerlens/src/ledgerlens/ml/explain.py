"""Plain-language reasons why an entry is on the review list."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ledgerlens.data.chart import account_name

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass(frozen=True)
class Reason:
    code: str
    text: str
    weight: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def eur(x: float) -> str:
    return f"€{x:,.2f}"


def hhmm(hours: float) -> str:
    h = int(hours) % 24
    return f"{h:02d}:{round((hours - int(hours)) * 60) % 60:02d}"


def explain(
    f: dict[str, Any], e: dict[str, Any], user_names: dict[str, str], limit: float = 10_000.0, top: int = 4
) -> list[Reason]:
    """``f`` is one row of features, ``e`` the matching journal entry."""
    name = user_names.get(str(e["created_by"]), str(e["created_by"]))
    amount = float(e["amount"])
    is_person = not f["is_system"]
    r: list[Reason] = []
    if f["self_approved"]:
        r.append(Reason("self_approved", f"Created and approved by the same person ({name}).", 6))
    if f["missing_approval"]:
        r.append(
            Reason(
                "missing_approval", f"{eur(amount)} is above the {eur(limit)} approval limit but has no approver.", 6
            )
        )
    if f["dup_reference"]:
        r.append(
            Reason(
                "dup_reference", f"Invoice reference {e['reference']} from this supplier is booked more than once.", 6
            )
        )
    if f["same_day_vendor_payments"] >= 2:
        r.append(
            Reason(
                "split",
                f"{int(f['same_day_vendor_payments']) + 1} payments to this supplier by {name} on the same day.",
                6,
            )
        )
    if f["dup_surprise"] > 0.5:
        r.append(
            Reason(
                "dup_amount",
                f"Same supplier and amount as {int(f['dup_vendor_amount'])} other invoice(s) within 14 days, which this supplier rarely has.",
                5,
            )
        )
    if is_person and f["user_hour_dev"] >= 3:
        r.append(
            Reason(
                "unusual_time",
                f"Posted at {str(e['posting_time'])[:5]}, while {name} usually posts around {hhmm(float(f['user_median_hour']))}.",
                5,
            )
        )
    if is_person and f["weekend_surprise"] >= 0.8:
        r.append(
            Reason(
                "unusual_day",
                f"Posted on a {WEEKDAYS[int(f['weekday'])]}, and {name} almost never posts at the weekend.",
                5,
            )
        )
    if f["poster_created_vendor"]:
        r.append(Reason("same_person_vendor", f"{name} created this supplier and also booked this entry.", 5))
    if f["pair_count"] < 10:
        pair = f"{account_name(e['dr_account'])} / {account_name(e['cr_account'])}"
        r.append(
            Reason(
                "rare_pair", f"The account combination {pair} is used only {int(f['pair_count'])} times this year.", 4
            )
        )
    elif is_person and f["user_pair_count"] <= 3 and f["user_entries"] >= 20:
        r.append(
            Reason(
                "rare_for_user",
                f"{name} booked this account combination only {int(f['user_pair_count'])} of {int(f['user_entries'])} times.",
                3,
            )
        )
    if f["below_limit"]:
        r.append(Reason("below_limit", f"{eur(amount)} is just under the {eur(limit)} approval limit.", 4))
    if f["new_vendor"]:
        r.append(
            Reason("new_vendor", f"The supplier was created {int(f['vendor_age_days'])} days before this entry.", 3)
        )
    if f["vague_description"]:
        r.append(Reason("vague", f'Vague description: "{e["description"]}".', 3))
    if f["manual_revenue"]:
        r.append(Reason("manual_revenue", "Manual entry to revenue.", 3))
    if f["manual_cash_out"]:
        r.append(Reason("manual_cash", "Manual journal that takes money out of the bank account.", 3))
    if f["quarter_end"] and f["is_manual"]:
        r.append(Reason("quarter_end", "Booked in the last days of the quarter.", 2))
    if f["reversed"]:
        r.append(Reason("reversed", "Reversed by an entry with the same amount within 10 days.", 2))
    if f["round_1000"]:
        r.append(Reason("round", f"Round amount of {eur(amount)}.", 2))
    if f["amount_z_in_pair"] >= 3:
        r.append(Reason("large", "The amount is unusually large for this account combination.", 2))
    r.sort(key=lambda x: -x.weight)
    return r[:top]
