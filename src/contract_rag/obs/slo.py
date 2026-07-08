"""SLO gate over S1's trace aggregates: turns P95-latency / error-rate / cost-per-request
into a gate-G1 pass/fail (spec §7). The eval-side surface of the latency/cost dashboards."""
from __future__ import annotations

from pydantic import BaseModel


class SLOThresholds(BaseModel):
    p95_ms: float = 30000.0
    error_rate: float = 0.05
    cost_per_request: float = 0.05


class SLOReport(BaseModel):
    passed: bool
    checks: dict[str, dict]


def _worst_p95(agg: dict) -> float:
    stages = agg.get("per_stage", {})
    return max((m["p95_ms"] for m in stages.values()), default=0.0)


def check_slo(agg: dict, thresholds: SLOThresholds | None = None) -> SLOReport:
    t = thresholds or SLOThresholds()
    measured = {
        "p95_ms": _worst_p95(agg),
        "error_rate": agg.get("error_rate", 0.0),
        "cost_per_request": agg.get("cost_per_request", 0.0),
    }
    limits = {"p95_ms": t.p95_ms, "error_rate": t.error_rate, "cost_per_request": t.cost_per_request}
    checks = {
        k: {"value": measured[k], "threshold": limits[k], "ok": measured[k] <= limits[k]}
        for k in measured
    }
    return SLOReport(passed=all(c["ok"] for c in checks.values()), checks=checks)


def format_slo(report: SLOReport) -> str:
    lines = [f"=== SLO gate: {'PASS' if report.passed else 'FAIL'} ==="]
    for name, c in report.checks.items():
        mark = "ok" if c["ok"] else "BREACH"
        lines.append(f"  {name:<18} {c['value']:.4f} <= {c['threshold']:.4f}  [{mark}]")
    return "\n".join(lines)
