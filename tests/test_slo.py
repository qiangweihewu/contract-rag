from contract_rag.api.slo import DEFAULT_SLO, SLO, SLOReport, check_slo


def _agg(p95_by_stage, error_rate=0.0, cost=0.0, n=3):
    return {
        "n_traces": n, "error_rate": error_rate, "cost_per_request": cost,
        "total_tokens": 0,
        "per_stage": {s: {"p95_ms": v, "p50_ms": 0.0, "error_rate": 0.0, "count": n}
                      for s, v in p95_by_stage.items()},
    }


def test_empty_run_meets_slo():
    rep = check_slo(_agg({}, n=0))
    assert isinstance(rep, SLOReport)
    assert rep.met is True
    assert rep.breaches == []


def test_within_budget_meets_slo():
    rep = check_slo(_agg({"parse": 100.0, "extract": 50.0}, error_rate=0.0, cost=0.0))
    assert rep.met is True
    assert rep.p95_latency_ms == 150.0


def test_latency_breach_fails_slo():
    slo = SLO(p95_latency_ms=100.0, error_rate=0.05, cost_per_request_usd=1.0)
    rep = check_slo(_agg({"parse": 200.0}), slo=slo)
    assert rep.met is False
    assert any("latency" in b for b in rep.breaches)


def test_error_rate_breach_fails_slo():
    slo = SLO(p95_latency_ms=10000.0, error_rate=0.01, cost_per_request_usd=1.0)
    rep = check_slo(_agg({"parse": 1.0}, error_rate=0.2), slo=slo)
    assert rep.met is False
    assert any("error_rate" in b for b in rep.breaches)


def test_default_slo_exists():
    assert DEFAULT_SLO.p95_latency_ms > 0
