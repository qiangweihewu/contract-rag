"""Instrumented baseline run: writes per-doc traces to JSONL and prints metrics.

Usage:
    EXTRACT_BACKEND=rule uv run python -m contract_rag.obs.run
"""
from __future__ import annotations

import os
from pathlib import Path

from contract_rag.baseline import run_baseline
from contract_rag.config import assert_backend_allowed, get_settings
from contract_rag.eval.ir_cache import ir_cache
from contract_rag.extract.extractor import get_extractor
from contract_rag.obs.metrics import aggregate_traces
from contract_rag.obs.report import format_metrics, render_trace
from contract_rag.obs.store import JsonlTraceStore
from contract_rag.obs.tracer import Tracer
from contract_rag.parse.docling_parser import parse_with_docling


def format_traces(traces) -> str:
    """Render every trace's span tree (latency / tokens / cost / status) for inspection."""
    return "\n\n".join(render_trace(t) for t in traces)


def main() -> None:
    settings = get_settings()
    assert_backend_allowed(settings)
    extractor = get_extractor(settings)
    parse_fn = ir_cache(Path(os.environ.get("IR_CACHE_DIR", ".ir_cache")), parse_with_docling)
    trace_out = os.environ.get("TRACE_OUT", "traces.jsonl")
    # Remove any stale trace file so the dashboard reflects only this run, not prior runs.
    Path(trace_out).unlink(missing_ok=True)
    store = JsonlTraceStore(trace_out)
    tracer = Tracer(store=store)
    run_baseline(settings, extractor, parse_fn, tracer=tracer)
    print(format_metrics(aggregate_traces(store.all())))
    if os.environ.get("TRACE_RENDER"):
        print()
        print(format_traces(store.all()))


if __name__ == "__main__":
    main()
