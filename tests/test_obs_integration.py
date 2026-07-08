from __future__ import annotations

import pytest

from contract_rag.baseline import run_baseline
from contract_rag.config import Settings
from contract_rag.ir import DocumentIR
from contract_rag.obs.models import SpanStatus
from contract_rag.obs.store import InMemoryTraceStore
from contract_rag.obs.tracer import Tracer


class _FakeGold:
    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.source_pdf = f"{doc_id}.pdf"


def _ir(doc_id):
    return DocumentIR(doc_id=doc_id, source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", blocks=[], metadata={})


class _FakeExtractor:
    def extract(self, ir):
        from contract_rag.extract.schema import ContractFacts, ExtractedClause
        return ContractFacts(
            counterparty=ExtractedClause(),
            effective_date=ExtractedClause(),
            governing_law=ExtractedClause(),
        )


def test_run_baseline_records_spans_per_doc(monkeypatch):
    import contract_rag.baseline as bl

    golden = [_FakeGold("a"), _FakeGold("b")]
    monkeypatch.setattr(bl, "load_golden_set", lambda _dir: golden)
    monkeypatch.setattr(bl, "row_for", lambda pred, g, ir, vertical=None: {"doc_id": g.doc_id})
    monkeypatch.setattr(bl, "aggregate", lambda rows, vertical=None: {"n_docs": len(rows)})

    store = InMemoryTraceStore()
    tracer = Tracer(store=store)
    out = run_baseline(
        Settings(), _FakeExtractor(), parse_fn=lambda p: _ir(p.stem), tracer=tracer
    )

    assert out == {"n_docs": 2}
    traces = store.all()
    assert [t.doc_id for t in traces] == ["a", "b"]
    assert [s.name for s in traces[0].spans] == ["parse", "extract"]


def test_run_baseline_without_tracer_still_works(monkeypatch):
    import contract_rag.baseline as bl
    monkeypatch.setattr(bl, "load_golden_set", lambda _dir: [_FakeGold("a")])
    monkeypatch.setattr(bl, "row_for", lambda pred, g, ir, vertical=None: {"doc_id": g.doc_id})
    monkeypatch.setattr(bl, "aggregate", lambda rows, vertical=None: {"n_docs": len(rows)})
    out = run_baseline(Settings(), _FakeExtractor(), parse_fn=lambda p: _ir(p.stem))
    assert out == {"n_docs": 1}


def test_run_baseline_instrumented_continues_on_error(monkeypatch):
    """Instrumented run: a failing first doc must NOT abort; both docs get persisted traces."""
    import contract_rag.baseline as bl

    call_count = 0

    def flaky_parse(p):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("parse boom")
        return _ir(p.stem)

    golden = [_FakeGold("a"), _FakeGold("b")]
    monkeypatch.setattr(bl, "load_golden_set", lambda _dir: golden)
    monkeypatch.setattr(bl, "row_for", lambda pred, g, ir, vertical=None: {"doc_id": g.doc_id})
    monkeypatch.setattr(bl, "aggregate", lambda rows, vertical=None: {"n_docs": len(rows)})

    store = InMemoryTraceStore()
    tracer = Tracer(store=store)
    # Must not raise — instrumented run continues past per-doc failure
    result = run_baseline(
        Settings(), _FakeExtractor(), parse_fn=flaky_parse, tracer=tracer
    )

    traces = store.all()
    assert len(traces) == 2, "both docs must produce a persisted trace"
    trace_a = next(t for t in traces if t.doc_id == "a")
    # First trace has an ERROR parse span
    parse_span = next(s for s in trace_a.spans if s.name == "parse")
    assert parse_span.status == SpanStatus.ERROR
    # Aggregate covers only surviving rows (doc "b")
    assert result == {"n_docs": 1}


def test_run_baseline_no_tracer_raises_on_error(monkeypatch):
    """No-tracer (default) path: exceptions propagate and abort the run as before."""
    import contract_rag.baseline as bl

    def _boom(p):
        raise RuntimeError("parse boom")

    monkeypatch.setattr(bl, "load_golden_set", lambda _dir: [_FakeGold("a")])
    monkeypatch.setattr(bl, "row_for", lambda pred, g, ir, vertical=None: {"doc_id": g.doc_id})
    monkeypatch.setattr(bl, "aggregate", lambda rows, vertical=None: {"n_docs": len(rows)})

    with pytest.raises(RuntimeError, match="parse boom"):
        run_baseline(Settings(), _FakeExtractor(), parse_fn=_boom)
