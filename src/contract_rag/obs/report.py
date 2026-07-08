from __future__ import annotations

from contract_rag.obs.models import Trace


def format_metrics(agg: dict) -> str:
    lines = [
        "=== contract-rag observability ===",
        f"traces: {agg['n_traces']}",
        f"error_rate:       {agg['error_rate']:.3f}",
        f"cost_per_request: ${agg['cost_per_request']:.5f}",
        f"total_tokens:     {agg['total_tokens']}",
        "per-stage latency (ms) / error-rate:",
    ]
    for stage, m in agg["per_stage"].items():
        lines.append(
            f"  {stage:<14} p50={m['p50_ms']:.1f}  p95={m['p95_ms']:.1f}  "
            f"err={m['error_rate']:.3f}  n={m['count']}"
        )
    return "\n".join(lines)


def render_trace(trace: Trace) -> str:
    lines = [f"trace {trace.trace_id} (doc {trace.doc_id}) total={trace.duration_ms:.1f}ms"]
    for s in trace.spans:
        suffix = f" [{s.status.value}:{s.error_type}]" if s.error_type else f" [{s.status.value}]"
        lines.append(f"  {s.name:<14} {s.duration_ms:.1f}ms tokens={s.tokens} ${s.cost_usd:.5f}{suffix}")
    return "\n".join(lines)
