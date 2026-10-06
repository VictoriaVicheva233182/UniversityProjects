# Methodology

## The ledger

Noordkade Logistics B.V. is fictional. Its financial year 2025 has about 198,000 journal entries from these processes: sales invoices (nightly batch, also on Saturdays), customer receipts (early morning bank import), supplier invoices, payment runs on Tuesday and Thursday, manual payments, payroll, depreciation, bank charges and month-end journals (accruals with reversals, reclasses, prepayments, inventory counts, year-end close at the weekend).

Legitimate entries were built to trip simple rules: system batches at night, round rent and lease amounts, weekly cleaning invoices with the same amount, accruals rounded to 500 euro, new suppliers created during the year, and a controller who sometimes approves his own accruals (a control weakness, but not fraud).

## The fraud schemes

| Scheme | Entries | Pattern |
|---|---:|---|
| Fictitious revenue at quarter end | 7 | Controller, late evening, round amounts, reversed next quarter |
| Payments split under the approval limit | 10 | Same clerk, same supplier, same day, just under 10,000 euro, no invoice |
| Ghost supplier | 14 | Clerk creates a supplier, books and pays its round invoices herself |
| Self-approved cash journals | 6 | Accountant approves his own journals that credit the bank |
| Night-time postings by the payroll officer | 6 | Weekend nights, payroll liabilities to bank |
| Expenses moved to fixed assets | 4 | Year-end reclass from costs to fixed assets |
| Duplicate supplier invoices | 6 | Booked again days later, half with a slightly changed reference |
| Cash to miscellaneous expenses | 6 | Vague descriptions such as "adj" and "see email" |

## Evaluation

Every method produces a ranked list. We measure:

- **Schemes found at k:** how many schemes have at least one entry in the first k entries. Finding one entry of a scheme is enough to start an investigation.
- **Entries to see every scheme:** the shortest list that contains all 8 schemes.
- **Fraud entries at k** and average precision, for completeness.

The classic rules are measured twice: as the full list of entries with at least one hit (what an auditor normally receives) and sorted by number of hits.

## Copilot evaluation

The copilot investigates the top 15 entries. We record whether each draft passes the number check, whether the model followed the protocol (or needed the fallback), how many tool calls it used, and, against the answers, whether it called fraud entries suspicious and named the right scheme.

## Limitations

The same person wrote the schemes and the detector, the data is synthetic, and the copilot's conclusions can be wrong even when its numbers are right. See the report for the full list.
