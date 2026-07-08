"""Business-metric instrumentation (spec §11.1/§25.4): AHT / time-saved, adoption,
CSAT-no-drop, cost-per-ticket, eval-pass-rate — the 'prove' phase's business story.
Pure aggregation over TicketEvent records (cf. obs/metrics.py). Credential-free."""
from __future__ import annotations

from statistics import mean
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from contract_rag.obs.models import Trace


class TicketEvent(BaseModel):
    ticket_id: str
    baseline_handle_seconds: float
    assisted_handle_seconds: float
    suggestion_offered: bool = True
    suggestion_accepted: bool = False
    csat_before: float | None = None
    csat_after: float | None = None
    llm_cost_usd: float = 0.0
    eval_passed: bool = True


class BusinessMetrics(BaseModel):
    n_tickets: int
    aht_seconds: float
    baseline_aht_seconds: float
    time_saved_seconds: float
    time_saved_pct: float
    adoption_rate: float
    cost_per_ticket: float
    csat_before: float | None
    csat_after: float | None
    csat_no_drop: bool
    eval_pass_rate: float


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def compute_business_metrics(events: list[TicketEvent]) -> BusinessMetrics:
    n = len(events)
    if n == 0:
        return BusinessMetrics(
            n_tickets=0, aht_seconds=0.0, baseline_aht_seconds=0.0, time_saved_seconds=0.0,
            time_saved_pct=0.0, adoption_rate=0.0, cost_per_ticket=0.0,
            csat_before=None, csat_after=None, csat_no_drop=True, eval_pass_rate=0.0,
        )
    aht = mean(e.assisted_handle_seconds for e in events)
    baseline_aht = mean(e.baseline_handle_seconds for e in events)
    saved = baseline_aht - aht
    offered = [e for e in events if e.suggestion_offered]
    adoption = (sum(1 for e in offered if e.suggestion_accepted) / len(offered)) if offered else 0.0
    csat_before = _mean_or_none([e.csat_before for e in events if e.csat_before is not None])
    csat_after = _mean_or_none([e.csat_after for e in events if e.csat_after is not None])
    no_drop = csat_before is None or csat_after is None or csat_after >= csat_before
    return BusinessMetrics(
        n_tickets=n,
        aht_seconds=aht,
        baseline_aht_seconds=baseline_aht,
        time_saved_seconds=saved,
        time_saved_pct=(saved / baseline_aht if baseline_aht else 0.0),
        adoption_rate=adoption,
        cost_per_ticket=mean(e.llm_cost_usd for e in events),
        csat_before=csat_before,
        csat_after=csat_after,
        csat_no_drop=no_drop,
        eval_pass_rate=mean(1.0 if e.eval_passed else 0.0 for e in events),
    )


class BusinessGate(BaseModel):
    min_adoption_rate: float = 0.40
    max_cost_per_ticket: float = 0.50
    require_csat_no_drop: bool = True


def check_business_gate(m: BusinessMetrics, gate: BusinessGate | None = None) -> dict:
    g = gate or BusinessGate()
    checks = {
        "adoption_rate": {
            "value": m.adoption_rate, "threshold": g.min_adoption_rate,
            "ok": m.adoption_rate >= g.min_adoption_rate,
        },
        "cost_per_ticket": {
            "value": m.cost_per_ticket, "threshold": g.max_cost_per_ticket,
            "ok": m.cost_per_ticket <= g.max_cost_per_ticket,
        },
        "csat_no_drop": {
            "value": 1.0 if m.csat_no_drop else 0.0, "threshold": 1.0,
            "ok": (m.csat_no_drop or not g.require_csat_no_drop),
        },
    }
    return {"passed": all(c["ok"] for c in checks.values()), "checks": checks}


def format_business(m: BusinessMetrics) -> str:
    csat_b = "n/a" if m.csat_before is None else f"{m.csat_before:.2f}"
    csat_a = "n/a" if m.csat_after is None else f"{m.csat_after:.2f}"
    return "\n".join([
        "=== business metrics ===",
        f"tickets:        {m.n_tickets}",
        f"AHT (assisted): {m.aht_seconds:.1f}s  (baseline {m.baseline_aht_seconds:.1f}s)",
        f"time_saved:     {m.time_saved_seconds:.1f}s  ({m.time_saved_pct:.1%})",
        f"adoption_rate:  {m.adoption_rate:.1%}",
        f"cost_per_ticket:${m.cost_per_ticket:.4f}",
        f"CSAT:           {csat_b} -> {csat_a}  (no_drop={m.csat_no_drop})",
        f"eval_pass_rate: {m.eval_pass_rate:.3f}",
    ])


def business_dashboard(events: list[TicketEvent], traces: list[Trace] | None = None) -> str:
    from contract_rag.obs.metrics import aggregate_traces
    from contract_rag.obs.report import format_metrics

    sections: list[str] = []
    if traces:
        sections.append(format_metrics(aggregate_traces(traces)))
    m = compute_business_metrics(events)
    sections.append(format_business(m))
    gate = check_business_gate(m)
    verdict = "PASS" if gate["passed"] else "FAIL"
    gate_lines = [f"=== gate G2 (business): {verdict} ==="]
    for name, c in gate["checks"].items():
        mark = "ok" if c["ok"] else "BREACH"
        gate_lines.append(f"  {name:<16} {c['value']:.4f} vs {c['threshold']:.4f}  [{mark}]")
    sections.append("\n".join(gate_lines))
    return "\n\n".join(sections)
