from contract_rag.obs.models import Span, SpanStatus, Trace


def test_trace_aggregates_over_spans():
    t = Trace(
        trace_id="t1",
        doc_id="d1",
        spans=[
            Span(name="parse", duration_ms=10.0, tokens=0, cost_usd=0.0),
            Span(name="extract", duration_ms=40.0, tokens=1000, cost_usd=0.01),
        ],
    )
    assert t.duration_ms == 50.0
    assert t.tokens == 1000
    assert round(t.cost_usd, 4) == 0.01
    assert t.ok is True


def test_trace_ok_false_when_any_span_errored():
    t = Trace(
        trace_id="t2",
        doc_id="d2",
        spans=[
            Span(name="parse", duration_ms=5.0),
            Span(name="extract", duration_ms=5.0, status=SpanStatus.ERROR, error_type="ValueError"),
        ],
    )
    assert t.ok is False
