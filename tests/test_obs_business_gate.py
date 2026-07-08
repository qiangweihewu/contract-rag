from contract_rag.obs.business import (
    BusinessGate,
    TicketEvent,
    check_business_gate,
    compute_business_metrics,
)


def _m(adoption_accepts, cost, csat_after):
    # 5 offered tickets; `adoption_accepts` of them accepted.
    events = [
        TicketEvent(ticket_id=f"t{i}", baseline_handle_seconds=100, assisted_handle_seconds=80,
                    suggestion_offered=True, suggestion_accepted=(i < adoption_accepts),
                    csat_before=4.0, csat_after=csat_after, llm_cost_usd=cost)
        for i in range(5)
    ]
    return compute_business_metrics(events)


def test_gate_passes_above_thresholds():
    res = check_business_gate(_m(adoption_accepts=3, cost=0.10, csat_after=4.1))  # adoption 0.6
    assert res["passed"] is True
    assert res["checks"]["adoption_rate"]["ok"] is True


def test_low_adoption_fails_gate():
    res = check_business_gate(_m(adoption_accepts=1, cost=0.10, csat_after=4.1))  # adoption 0.2
    assert res["passed"] is False
    assert res["checks"]["adoption_rate"]["ok"] is False


def test_csat_drop_fails_gate():
    res = check_business_gate(_m(adoption_accepts=3, cost=0.10, csat_after=3.5))
    assert res["passed"] is False
    assert res["checks"]["csat_no_drop"]["ok"] is False
