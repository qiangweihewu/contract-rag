from contract_rag.obs.slo import SLOThresholds, check_slo, format_slo


def _agg(p95, error_rate, cost):
    return {"n_traces": 3, "error_rate": error_rate, "cost_per_request": cost,
            "total_tokens": 0,
            "per_stage": {"parse": {"count": 3, "p50_ms": 1.0, "p95_ms": p95 / 2, "error_rate": 0.0},
                          "extract": {"count": 3, "p50_ms": 2.0, "p95_ms": p95, "error_rate": error_rate}}}


def test_all_within_threshold_passes():
    report = check_slo(_agg(p95=100.0, error_rate=0.0, cost=0.001))
    assert report.passed is True
    assert report.checks["p95_ms"]["ok"] is True
    assert report.checks["p95_ms"]["value"] == 100.0  # worst per-stage p95


def test_breach_fails_and_flags_the_metric():
    report = check_slo(_agg(p95=100.0, error_rate=0.5, cost=0.001),
                       SLOThresholds(error_rate=0.05))
    assert report.passed is False
    assert report.checks["error_rate"]["ok"] is False


def test_format_slo_reports_pass_fail():
    out = format_slo(check_slo(_agg(100.0, 0.0, 0.001)))
    assert "PASS" in out or "pass" in out.lower()
