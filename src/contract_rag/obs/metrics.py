from __future__ import annotations

import math

from contract_rag.obs.models import SpanStatus, Trace


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def aggregate_traces(traces: list[Trace]) -> dict:
    n = len(traces)
    if n == 0:
        return {
            "n_traces": 0,
            "error_rate": 0.0,
            "cost_per_request": 0.0,
            "total_tokens": 0,
            "per_stage": {},
        }

    by_stage: dict[str, list[tuple[float, bool]]] = {}
    for t in traces:
        for span in t.spans:
            by_stage.setdefault(span.name, []).append(
                (span.duration_ms, span.status != SpanStatus.OK)
            )

    per_stage: dict[str, dict] = {}
    for stage, rows in by_stage.items():
        durations = [d for d, _ in rows]
        errors = sum(1 for _, e in rows if e)
        per_stage[stage] = {
            "count": len(rows),
            "p50_ms": percentile(durations, 0.5),
            "p95_ms": percentile(durations, 0.95),
            "error_rate": round(errors / len(rows), 10),
        }

    return {
        "n_traces": n,
        "error_rate": sum(0 if t.ok else 1 for t in traces) / n,
        "cost_per_request": sum(t.cost_usd for t in traces) / n,
        "total_tokens": sum(t.tokens for t in traces),
        "per_stage": per_stage,
    }
