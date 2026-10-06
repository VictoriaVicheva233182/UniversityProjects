"""Classic journal entry tests, as auditors run them today.

Each rule is a yes or no test. In practice an auditor gets every entry that
hits at least one rule, which is usually thousands of entries.
"""

from __future__ import annotations

import pandas as pd

RULES: dict[str, str] = {
    "weekend_or_holiday": "Posted on a weekend or public holiday",
    "outside_business_hours": "Posted before 07:00 or after 20:00",
    "round_amount": "Round amount (multiple of 1,000)",
    "just_below_limit": "Just below the approval limit",
    "manual_revenue": "Manual entry to revenue",
    "self_approved": "Created and approved by the same user",
    "missing_approval": "Above the approval limit without an approver",
    "vague_description": "Vague or missing description",
    "period_end_manual": "Manual entry in the last 3 days of a month",
    "rare_account_pair": "Account combination used fewer than 10 times",
    "duplicate_invoice": "Same supplier and amount within 14 days",
    "duplicate_reference": "Same supplier and invoice reference booked twice",
    "new_supplier": "Invoice from a supplier created less than 60 days ago",
}


def run_rules(f: pd.DataFrame) -> pd.DataFrame:
    hits = pd.DataFrame(index=f.index)
    # Time based tests skip scheduled system batches, as auditors normally do.
    by_person = f["is_system"] == 0
    hits["weekend_or_holiday"] = (f["weekend_or_holiday"] == 1) & by_person
    hits["outside_business_hours"] = (f["off_hours"] == 1) & by_person
    hits["round_amount"] = f["round_1000"] == 1
    hits["just_below_limit"] = f["below_limit"] == 1
    hits["manual_revenue"] = f["manual_revenue"] == 1
    hits["self_approved"] = f["self_approved"] == 1
    hits["missing_approval"] = f["missing_approval"] == 1
    hits["vague_description"] = f["vague_description"] == 1
    hits["period_end_manual"] = (f["is_manual"] == 1) & (f["days_to_month_end"] <= 2)
    hits["rare_account_pair"] = f["pair_count"] < 10
    hits["duplicate_invoice"] = f["dup_vendor_amount"] > 0
    hits["duplicate_reference"] = f["dup_reference"] > 0
    hits["new_supplier"] = f["new_vendor"] == 1
    return hits.astype(bool)


def rule_summary(hits: pd.DataFrame) -> pd.Series:
    return hits.apply(lambda row: "|".join(k for k, v in row.items() if v), axis=1)
