from __future__ import annotations

from contract_rag.obs.run import format_traces
from contract_rag.obs.store import InMemoryTraceStore
from contract_rag.obs.tracer import Tracer


def test_format_traces_renders_each_trace():
    store = InMemoryTraceStore()
    tracer = Tracer(store=store)
    trace = tracer.start(doc_id="doc-1")
    with tracer.span(trace, "parse"):
        pass
    tracer.finish(trace)

    out = format_traces(store.all())
    assert "doc-1" in out
    assert "parse" in out
    assert trace.trace_id in out


def test_format_traces_empty_is_empty_string():
    assert format_traces([]) == ""
