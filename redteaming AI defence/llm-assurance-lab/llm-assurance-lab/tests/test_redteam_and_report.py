from pathlib import Path

from assurance_lab.redteam.attacks import load_benign, load_suite
from assurance_lab.redteam.runner import expand_cases, load_run, run_profile, save_run
from assurance_lab.redteam.scoring import wilson_interval
from assurance_lab.reporting.report import write_report
from assurance_lab.reporting.risk import rate_risk

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "data/attacks/attack_suite.yaml"
BENIGN = ROOT / "data/benign/benign_eval.yaml"


def test_suite_is_valid_and_covers_every_category():
    suite = load_suite(SUITE)
    used = {c.category for c in suite.cases}
    assert used == set(suite.categories)
    assert len(expand_cases(suite, ["roleplay"])) > len(suite.cases)


def test_wilson_interval_bounds():
    assert wilson_interval(0, 0) == (0.0, 0.0)
    low, high = wilson_interval(5, 10)
    assert 0.2 < low < 0.5 < high < 0.8


def test_risk_matrix():
    assert rate_risk(5, 0.0).rating == "Low"
    assert rate_risk(5, 0.08).rating == "High"
    assert rate_risk(5, 0.9).rating == "Critical"
    assert rate_risk(3, 0.3).rating == "High"


def test_end_to_end_run_and_report(baseline, hardened, tmp_path: Path):
    suite, benign = load_suite(SUITE), load_benign(BENIGN)
    b_out = run_profile(baseline, suite, benign, converters=["roleplay"])
    h_out = run_profile(hardened, suite, benign, converters=["roleplay"])
    assert b_out.summary["attacks"]["asr"] > h_out.summary["attacks"]["asr"]
    assert b_out.summary["simulated"] is True

    b_dir, h_dir = save_run(b_out, tmp_path / "runs"), save_run(h_out, tmp_path / "runs")
    summary, attacks, benign_rows = load_run(b_dir)
    assert summary["attacks"]["runs"] == len(attacks) and len(benign_rows) == len(benign.questions)

    html, md = write_report(b_dir, h_dir, tmp_path / "report")
    text = html.read_text()
    assert "AI assurance report" in text and "Simulated run" in text
    assert "F-01" in md.read_text()
    assert "\u2014" not in text  # house style: no em dashes
