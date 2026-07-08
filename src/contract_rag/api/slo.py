"""Service-level objectives for the contract-rag API, and a pure check that
grades an obs.metrics.aggregate_traces(...) dict against them. The numeric
targets are credential-free-pipeline defaults (rule extractor, hashing embedder)
and are re-anchored to a partner's real workflow during deployment (spec §7)."""
from __future__ import annotations

from pydantic import BaseModel


class SLO(BaseModel):
    p95_latency_ms: float = 5000.0          # end-to-end P95 per request
    error_rate: float = 0.02                # ≤ 2% of requests error/timeout
    cost_per_request_usd: float = 0.05      # mean external-LLM spend per request


DEFAULT_SLO = SLO()


class SLOReport(BaseModel):
    met: bool
    p95_latency_ms: float
    error_rate: float
    cost_per_request_usd: float
    breaches: list[str]


def check_slo(agg: dict, slo: SLO = DEFAULT_SLO) -> SLOReport:
    per_stage = agg.get("per_stage", {})
    p95 = sum(s.get("p95_ms", 0.0) for s in per_stage.values())
    error_rate = agg.get("error_rate", 0.0)
    cost = agg.get("cost_per_request", 0.0)
    breaches: list[str] = []
    if agg.get("n_traces", 0) > 0:
        if p95 > slo.p95_latency_ms:
            breaches.append(f"p95_latency {p95:.0f}ms > {slo.p95_latency_ms:.0f}ms")
        if error_rate > slo.error_rate:
            breaches.append(f"error_rate {error_rate:.3f} > {slo.error_rate:.3f}")
        if cost > slo.cost_per_request_usd:
            breaches.append(f"cost_per_request ${cost:.5f} > ${slo.cost_per_request_usd:.5f}")
    return SLOReport(
        met=not breaches, p95_latency_ms=p95, error_rate=error_rate,
        cost_per_request_usd=cost, breaches=breaches,
    )
