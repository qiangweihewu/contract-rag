from contract_rag.obs.models import Span, Trace
from contract_rag.obs.store import InMemoryTraceStore, JsonlTraceStore


def _trace(tid: str) -> Trace:
    return Trace(trace_id=tid, doc_id="d", spans=[Span(name="parse", duration_ms=1.0)])


def test_in_memory_store_roundtrip():
    store = InMemoryTraceStore()
    store.add(_trace("a"))
    store.add(_trace("b"))
    assert [t.trace_id for t in store.all()] == ["a", "b"]


def test_jsonl_store_persists_and_reads_back(tmp_path):
    path = tmp_path / "traces.jsonl"
    store = JsonlTraceStore(path)
    store.add(_trace("a"))
    store.add(_trace("b"))

    reloaded = JsonlTraceStore(path).all()
    assert [t.trace_id for t in reloaded] == ["a", "b"]
    assert reloaded[0].spans[0].name == "parse"


def test_jsonl_store_all_is_empty_when_file_absent(tmp_path):
    assert JsonlTraceStore(tmp_path / "missing.jsonl").all() == []
