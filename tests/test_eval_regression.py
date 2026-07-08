from contract_rag.eval.regression import (
    format_regression,
    ragas_available,
    run_regression,
)


def test_no_regression_when_metrics_hold_or_improve():
    report = run_regression(current={"field_f1": 0.70, "source_accuracy": 1.0},
                            baseline={"field_f1": 0.68, "source_accuracy": 1.0})
    assert report.passed is True
    assert all(c.ok for c in report.checks)


def test_regression_flagged_when_metric_drops_beyond_tolerance():
    report = run_regression(current={"field_f1": 0.50},
                            baseline={"field_f1": 0.68},
                            tolerances={"field_f1": 0.02})
    assert report.passed is False
    bad = [c for c in report.checks if c.name == "field_f1"][0]
    assert bad.ok is False
    assert round(bad.delta, 3) == -0.18


def test_within_tolerance_is_not_a_regression():
    report = run_regression(current={"field_f1": 0.67},
                            baseline={"field_f1": 0.68},
                            tolerances={"field_f1": 0.02})
    assert report.passed is True


def test_ragas_available_is_a_bool():
    assert isinstance(ragas_available(), bool)


def test_format_regression_reports_verdict():
    out = format_regression(run_regression({"field_f1": 0.7}, {"field_f1": 0.68}))
    assert "regression" in out.lower()
