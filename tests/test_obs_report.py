# tests/test_obs_report.py
from contract_rag.obs.metrics import aggregate_traces
from contract_rag.obs.models import Span, SpanStatus, Trace
from contract_rag.obs.report import format_metrics, render_trace


def test_format_metrics_mentions_stages_and_totals():
    traces = [
        Trace(trace_id="t1", doc_id="d1", spans=[Span(name="parse", duration_ms=10.0)]),
        Trace(trace_id="t2", doc_id="d2", spans=[Span(name="parse", duration_ms=30.0)]),
    ]
    out = format_metrics(aggregate_traces(traces))
    assert "parse" in out
    assert "p95" in out.lower()
    assert "traces: 2" in out.lower() or "n_traces" in out.lower()


def test_render_trace_lists_each_span_with_status():
    trace = Trace(
        trace_id="t1", doc_id="d1",
        spans=[
            Span(name="parse", duration_ms=10.0),
            Span(name="extract", duration_ms=5.0, status=SpanStatus.ERROR, error_type="ValueError"),
        ],
    )
    out = render_trace(trace)
    assert "d1" in out
    assert "parse" in out and "extract" in out
    assert "ValueError" in out
