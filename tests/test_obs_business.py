from contract_rag.obs.business import TicketEvent, compute_business_metrics


def test_time_saved_and_adoption_and_cost():
    events = [
        TicketEvent(ticket_id="t1", baseline_handle_seconds=600, assisted_handle_seconds=300,
                    suggestion_offered=True, suggestion_accepted=True,
                    csat_before=4.0, csat_after=4.2, llm_cost_usd=0.02),
        TicketEvent(ticket_id="t2", baseline_handle_seconds=600, assisted_handle_seconds=300,
                    suggestion_offered=True, suggestion_accepted=False,
                    csat_before=4.0, csat_after=4.0, llm_cost_usd=0.04),
    ]
    m = compute_business_metrics(events)
    assert m.n_tickets == 2
    assert m.aht_seconds == 300.0
    assert m.baseline_aht_seconds == 600.0
    assert m.time_saved_seconds == 300.0
    assert m.time_saved_pct == 0.5
    assert m.adoption_rate == 0.5            # 1 accepted of 2 offered
    assert round(m.cost_per_ticket, 4) == 0.03
    assert m.csat_no_drop is True


def test_csat_drop_is_flagged():
    events = [TicketEvent(ticket_id="t", baseline_handle_seconds=100, assisted_handle_seconds=90,
                          csat_before=4.5, csat_after=4.0)]
    assert compute_business_metrics(events).csat_no_drop is False


def test_adoption_ignores_unoffered_tickets():
    events = [
        TicketEvent(ticket_id="a", baseline_handle_seconds=100, assisted_handle_seconds=80,
                    suggestion_offered=False),
        TicketEvent(ticket_id="b", baseline_handle_seconds=100, assisted_handle_seconds=80,
                    suggestion_offered=True, suggestion_accepted=True),
    ]
    assert compute_business_metrics(events).adoption_rate == 1.0   # 1 of 1 offered


def test_empty_events_are_safe():
    m = compute_business_metrics([])
    assert m.n_tickets == 0
    assert m.adoption_rate == 0.0
    assert m.csat_no_drop is True
