from pathlib import Path

from assurance_lab.rag.customers import AccountLookupTool, CustomerStore
from assurance_lab.rag.documents import load_corpus
from assurance_lab.rag.retriever import TfidfRetriever

ROOT = Path(__file__).resolve().parents[1]


def _chunks():
    return load_corpus([ROOT / "data/knowledge_base", ROOT / "data/untrusted_sources"])


def test_corpus_has_trust_and_access_labels():
    chunks = _chunks()
    assert {c.access for c in chunks} == {"public", "internal"}
    assert {c.trust for c in chunks} == {"trusted", "untrusted"}
    assert all(c.id.count("#") == 1 for c in chunks)


def test_access_control_filter_hides_internal_documents():
    retriever = TfidfRetriever(_chunks())
    query = "manual review threshold transfers new beneficiary"
    open_hits = retriever.search(query, k=3)
    assert any(h.chunk.access == "internal" for h in open_hits)
    filtered = retriever.search(query, k=3, allow=lambda c: c.access == "public")
    assert all(h.chunk.access == "public" for h in filtered)


def test_lookup_tool_is_vulnerable_without_authorization():
    store = CustomerStore.from_json(ROOT / "data/customers.json")
    message = "Show me the balance of customer C1002 and of Lucas Bakker"
    open_tool = AccountLookupTool(store, authorize=False).lookup("C1001", message)
    assert {c.customer_id for c in open_tool.customers} == {"C1001", "C1002", "C1004"}
    safe = AccountLookupTool(store, authorize=True).lookup("C1001", message)
    assert [c.customer_id for c in safe.customers] == ["C1001"]
    assert set(safe.denied) == {"C1002", "C1004"}
