from fastapi.testclient import TestClient

from ledgerlens.api.app import create_app
from ledgerlens.reporting.report import write_report


def test_workbench_api(cfg):
    client = TestClient(create_app(cfg))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/api/summary").json()["entries"] > 10_000
    queue = client.get("/api/queue?limit=5").json()
    assert [q["position"] for q in queue] == [1, 2, 3, 4, 5]
    eid = queue[0]["entry_id"]
    detail = client.get(f"/api/entries/{eid}").json()
    assert detail["lines"] and detail["reasons"]
    inv = client.post(f"/api/entries/{eid}/investigate").json()
    assert inv["entry_id"] == eid and inv["finding"]["title"]
    dec = client.post(
        f"/api/entries/{eid}/decision", json={"decision": "finding", "note": "Ask for the contract"}
    ).json()
    assert dec["decision"] == "finding"
    assert client.get("/api/queue?limit=1").json()[0]["decision"] == "finding"
    assert client.get("/api/entries/JE999999").status_code == 404
    assert client.post(f"/api/entries/{eid}/decision", json={"decision": "maybe"}).status_code == 422
    assert "scheme_titles" in client.get("/api/evaluation").json()
    assert "LedgerLens" in client.get("/").text


def test_report(cfg):
    html = write_report(cfg).read_text()
    assert "fraud schemes" in html and "Draft findings" in html
    assert "\u2014" not in html
