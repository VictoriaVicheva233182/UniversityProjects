from collections.abc import Sequence

from ledgerlens.copilot.agent import Copilot, parse_json
from ledgerlens.copilot.grounding import check_grounding, normalize_number
from ledgerlens.copilot.tools import LedgerTools, ToolError
from ledgerlens.llm.base import LLMClient, Message
from ledgerlens.llm.simulated import SimulatedLLM


def test_number_normalisation():
    assert normalize_number("9,850.00") == normalize_number("9850.0") == "9850"
    assert normalize_number("1,75") == "1.75"


def test_grounding_flags_invented_numbers_and_ids():
    evidence = '{"amount_eur": 9850.0, "date": "2025-10-15", "time": "16:09:12", "entry_id": "JE000123"}'
    ok = check_grounding("Paid €9,850.00 on 2025-10-15 at 16:09 (JE000123).", evidence)
    assert ok.grounded
    bad = check_grounding("Paid €12,400.00, see JE999999.", evidence)
    assert not bad.grounded
    assert "12400" in bad.ungrounded_numbers and "JE999999" in bad.unknown_ids


def test_parse_json_tolerates_fences_and_chatter():
    assert parse_json('Sure!\n```json\n{"action": "tool", "tool": "get_entry", "args": {}}\n```') == {
        "action": "tool",
        "tool": "get_entry",
        "args": {},
    }
    assert parse_json("no json here") is None


def test_tools_reject_unknown_tool_and_entry(ws):
    tools = LedgerTools(ws)
    for name, args in (("drop_table", {}), ("get_entry", {"entry_id": "JE999999"})):
        try:
            tools.call(name, args)
        except ToolError:
            continue
        raise AssertionError(f"{name} should fail")


def test_simulated_investigation_is_grounded(ws):
    eid = ws.scores.sort_values("rank").iloc[0]["entry_id"]
    inv = Copilot(ws, SimulatedLLM()).investigate(eid)
    assert inv.grounding["grounded"], inv.grounding
    assert [s.tool for s in inv.steps][:2] == ["get_entry", "user_profile"]
    assert not inv.fallback_used


class BrokenLLM(LLMClient):
    name = "broken"

    def __init__(self) -> None:
        super().__init__("broken")

    def chat(self, messages: Sequence[Message], json_mode: bool = False) -> str:
        return "I think this looks fine!"


def test_fallback_when_model_does_not_follow_protocol(ws):
    eid = ws.scores.sort_values("rank").iloc[0]["entry_id"]
    inv = Copilot(ws, BrokenLLM()).investigate(eid)
    assert inv.fallback_used
    assert inv.finding["assessment"] == "needs_more_evidence"
    assert inv.grounding["grounded"]
