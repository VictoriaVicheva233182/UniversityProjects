import pandas as pd

from ledgerlens.data.generator import generate_ledger
from ledgerlens.data.lines import entry_lines
from ledgerlens.data.schemes import SCHEMES


def test_generator_is_deterministic_and_plants_every_scheme():
    a = generate_ledger(3, 0.05)
    b = generate_ledger(3, 0.05)
    pd.testing.assert_frame_equal(a.entries, b.entries)
    assert set(a.ground_truth["scheme"]) == set(SCHEMES)
    assert a.entries["entry_id"].is_unique
    assert "_scheme" not in a.entries.columns


def test_every_entry_balances(ws):
    sample = ws.ledger.entries.sample(500, random_state=1).to_dict("records")
    for e in sample:
        lines = entry_lines(e)
        assert round(sum(x["debit"] for x in lines), 2) == round(sum(x["credit"] for x in lines), 2)
