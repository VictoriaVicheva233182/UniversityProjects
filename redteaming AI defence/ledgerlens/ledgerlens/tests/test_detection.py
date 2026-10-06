from ledgerlens.audit.rules import run_rules


def test_time_rules_skip_system_batches(ws):
    hits = run_rules(ws.features)
    system = ws.features["is_system"] == 1
    assert not hits.loc[system, "outside_business_hours"].any()
    assert hits["outside_business_hours"].sum() > 0


def test_exact_duplicate_references_are_caught(ws):
    truth = ws.ledger.ground_truth
    dups = truth.loc[truth["scheme"] == "duplicate_invoices", "entry_id"]
    flagged = [ws.feature_row(e)["dup_reference"] > 0 for e in dups]
    assert sum(flagged) >= 3


def test_metrics_cover_all_methods(cfg):
    import json

    m = json.loads((cfg.out / "metrics.json").read_text())
    assert set(m["methods"]) == {"score_rules", "score_isolation_forest", "score_autoencoder", "score_ledgerlens"}
    ll = m["methods"]["score_ledgerlens"]
    assert next(a for a in ll["at_k"] if a["k"] == 500)["schemes_found"] >= 6
    assert m["rules_any_hit"]["flagged_entries"] < m["population"] * 0.1
    assert m["benford_purchase_invoices"]["verdict"]


def test_reasons_are_plain_sentences(ws):
    top = ws.scores.sort_values("rank").head(10)["entry_id"]
    for eid in top:
        reasons = ws.reasons(eid)
        assert reasons, eid
        assert all(r.text.endswith(".") for r in reasons)
