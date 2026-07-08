from contract_rag.obs.metrics import aggregate_traces, percentile
from contract_rag.obs.models import Span, SpanStatus, Trace


def test_percentile_interpolates_and_handles_empty():
    assert percentile([], 0.5) == 0.0
    assert percentile([10.0], 0.95) == 10.0
    assert percentile([0.0, 10.0], 0.5) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5


def _trace(tid, parse_ms, extract_ms, extract_err=False, tokens=0, cost=0.0):
    return Trace(
        trace_id=tid, doc_id=tid,
        spans=[
            Span(name="parse", duration_ms=parse_ms),
            Span(
                name="extract", duration_ms=extract_ms, tokens=tokens, cost_usd=cost,
                status=SpanStatus.ERROR if extract_err else SpanStatus.OK,
            ),
        ],
    )


def test_aggregate_traces_reports_per_stage_and_cost():
    traces = [
        _trace("t1", 10.0, 100.0, tokens=1000, cost=0.005),
        _trace("t2", 20.0, 200.0, tokens=2000, cost=0.010),
        _trace("t3", 30.0, 300.0, extract_err=True),
    ]
    agg = aggregate_traces(traces)
    assert agg["n_traces"] == 3
    assert round(agg["error_rate"], 3) == round(1 / 3, 3)        # t3 errored
    assert round(agg["cost_per_request"], 4) == 0.005            # (0.005+0.010+0)/3
    assert agg["total_tokens"] == 3000
    assert agg["per_stage"]["parse"]["count"] == 3
    assert agg["per_stage"]["parse"]["p50_ms"] == 20.0
    assert agg["per_stage"]["extract"]["error_rate"] == round(1 / 3, 10)


def test_aggregate_traces_empty_is_safe():
    agg = aggregate_traces([])
    assert agg["n_traces"] == 0
    assert agg["error_rate"] == 0.0
    assert agg["per_stage"] == {}
