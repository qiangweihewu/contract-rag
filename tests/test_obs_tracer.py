import pytest

from contract_rag.obs.models import SpanStatus
from contract_rag.obs.store import InMemoryTraceStore  # defined in Task 3
from contract_rag.obs.tracer import NoopTracer, Tracer


def _fake_clock():
    """Deterministic clock: returns 0, 1, 2, ... seconds on each call."""
    ticks = iter(range(1000))
    return lambda: float(next(ticks))


def _ids():
    seq = iter(["trace-0", "trace-1", "trace-2"])
    return lambda: next(seq)


def test_span_times_block_and_persists_trace():
    store = InMemoryTraceStore()
    tracer = Tracer(store=store, clock=_fake_clock(), id_factory=_ids())
    trace = tracer.start(doc_id="d1")
    with tracer.span(trace, "parse"):
        pass
    tracer.finish(trace)

    assert trace.trace_id == "trace-0"
    assert len(trace.spans) == 1
    assert trace.spans[0].name == "parse"
    assert trace.spans[0].duration_ms == 1000.0  # (1 - 0)s * 1000
    assert trace.spans[0].status == SpanStatus.OK
    assert store.all() == [trace]


def test_span_records_error_and_reraises():
    tracer = Tracer(store=InMemoryTraceStore(), clock=_fake_clock(), id_factory=_ids())
    trace = tracer.start(doc_id="d2")
    with pytest.raises(ValueError):
        with tracer.span(trace, "extract"):
            raise ValueError("boom")
    assert trace.spans[0].status == SpanStatus.ERROR
    assert trace.spans[0].error_type == "ValueError"


def test_span_records_timeout_distinctly():
    tracer = Tracer(store=InMemoryTraceStore(), clock=_fake_clock(), id_factory=_ids())
    trace = tracer.start(doc_id="d3")
    with pytest.raises(TimeoutError):
        with tracer.span(trace, "extract"):
            raise TimeoutError()
    assert trace.spans[0].status == SpanStatus.TIMEOUT


def test_noop_tracer_is_a_safe_noop():
    tracer = NoopTracer()
    trace = tracer.start(doc_id="d")
    with tracer.span(trace, "parse") as span:
        span.tokens = 5  # caller may set fields; discarded
    tracer.finish(trace)  # no store, no error
    assert trace.spans == []
