"""Regression harness (F1): compare current eval metrics to a saved baseline with
per-metric tolerances; fail if any metric regressed. The default metric source is the
project's own eval (field-F1, source-accuracy, Context-Recall) — credential-free.
deepeval/ragas are OPTIONAL adapters (`[eval]` extra), lazy-checked here so the default
path needs neither installed."""
from __future__ import annotations

import importlib.util

from pydantic import BaseModel


class MetricCheck(BaseModel):
    name: str
    baseline: float
    current: float
    delta: float
    tolerance: float
    ok: bool


class RegressionReport(BaseModel):
    passed: bool
    checks: list[MetricCheck]


def run_regression(
    current: dict[str, float],
    baseline: dict[str, float],
    tolerances: dict[str, float] | None = None,
    default_tolerance: float = 0.0,
) -> RegressionReport:
    tol = tolerances or {}
    checks: list[MetricCheck] = []
    for name, base in baseline.items():
        cur = current.get(name, 0.0)
        t = tol.get(name, default_tolerance)
        delta = cur - base
        checks.append(MetricCheck(
            name=name, baseline=base, current=cur, delta=delta, tolerance=t,
            ok=cur >= base - t,
        ))
    return RegressionReport(passed=all(c.ok for c in checks), checks=checks)


def format_regression(report: RegressionReport) -> str:
    lines = [f"=== regression gate: {'PASS (no regression)' if report.passed else 'REGRESSION'} ==="]
    for c in report.checks:
        mark = "ok" if c.ok else "REGRESSED"
        lines.append(f"  {c.name:<18} {c.baseline:.3f} -> {c.current:.3f} ({c.delta:+.3f})  [{mark}]")
    return "\n".join(lines)


def ragas_available() -> bool:
    return importlib.util.find_spec("ragas") is not None


def deepeval_available() -> bool:
    return importlib.util.find_spec("deepeval") is not None
