from contract_rag.obs.business import (
    TicketEvent,
    business_dashboard,
    compute_business_metrics,
    format_business,
)
from contract_rag.obs.models import Span, Trace


def _events():
    return [TicketEvent(ticket_id="t", baseline_handle_seconds=600, assisted_handle_seconds=300,
                        suggestion_offered=True, suggestion_accepted=True,
                        csat_before=4.0, csat_after=4.1, llm_cost_usd=0.02)]


def test_format_business_mentions_key_metrics():
    out = format_business(compute_business_metrics(_events()))
    low = out.lower()
    assert "aht" in low or "handle" in low
    assert "adoption" in low
    assert "csat" in low


def test_dashboard_includes_technical_section_when_traces_given():
    traces = [Trace(trace_id="t1", doc_id="d1", spans=[Span(name="parse", duration_ms=10.0)])]
    out = business_dashboard(_events(), traces=traces)
    assert "parse" in out          # S1 technical section present
    assert "adoption" in out.lower()  # business section present
    assert "gate" in out.lower()      # gate verdict present


def test_dashboard_without_traces_still_shows_business():
    out = business_dashboard(_events())
    assert "adoption" in out.lower()
