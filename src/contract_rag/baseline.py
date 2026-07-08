from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from contract_rag.verticals.base import Vertical

from contract_rag.config import Settings, assert_backend_allowed, get_settings
from contract_rag.eval.golden import load_golden_set
from contract_rag.eval.metrics import aggregate, row_for
from contract_rag.ir import DocumentIR


def run_baseline(
    settings: Settings, extractor, parse_fn: Callable[[Path], DocumentIR],
    *, tracer=None, vertical: Vertical | None = None, collect_rows: list | None = None,
) -> dict:
    from contract_rag.obs.tracer import NoopTracer
    from contract_rag.verticals.registry import get_vertical_for

    vertical = vertical or get_vertical_for(settings)
    instrumented = tracer is not None
    tracer = tracer or NoopTracer()
    golden = load_golden_set(settings.golden_set_dir)
    if not golden:
        raise ValueError(
            f"golden set is empty: no *.json found in {settings.golden_set_dir}; "
            "run build_golden_from_cuad(...) first"
        )
    rows = []
    for g in golden:
        trace = tracer.start(doc_id=g.doc_id)
        try:
            with tracer.span(trace, "parse"):
                ir = parse_fn(settings.data_dir / g.source_pdf)
            with tracer.span(trace, "extract") as span:
                pred = extractor.extract(ir)
                span.tokens = getattr(extractor, "last_tokens", 0)
                span.cost_usd = getattr(extractor, "last_cost_usd", 0.0)
            rows.append(row_for(pred, g, ir, vertical))
        except Exception:
            if not instrumented:
                raise
            # instrumented: the error/timeout span is already recorded on the trace;
            # persist it (finally) and continue so one bad doc can't hide the error-rate.
        finally:
            tracer.finish(trace)
    if collect_rows is not None:
        collect_rows.extend(rows)
    return aggregate(rows, vertical)


def format_report(agg: dict) -> str:
    lines = [
        "=== Contract-RAG Phase 0 baseline (CUAD) ===",
        f"docs:            {agg['n_docs']}",
        f"field_f1:        {agg['field_f1']:.3f}",
        f"precision:       {agg['precision']:.3f}",
        f"recall:          {agg['recall']:.3f}",
        f"source_accuracy: {agg['source_accuracy']:.3f}",
        "per-field accuracy:",
    ]
    for name, acc in agg["per_field"].items():
        lines.append(f"  {name:<18} {acc:.3f}")
    return "\n".join(lines)


def main() -> None:
    import os

    from contract_rag.eval.ir_cache import ir_cache
    from contract_rag.extract.extractor import get_extractor
    from contract_rag.parse.docling_parser import parse_with_docling

    settings = get_settings()
    assert_backend_allowed(settings)
    from contract_rag.verticals.registry import get_vertical_for
    vertical = get_vertical_for(settings)
    extractor = get_extractor(settings, vertical)
    # Cache parsed IRs (docling is the slow step) so re-runs across backends are fast.
    parse_fn = ir_cache(Path(os.environ.get("IR_CACHE_DIR", ".ir_cache")), parse_with_docling)
    stats_ci = bool(os.environ.get("STATS_CI"))
    collect_rows: list | None = [] if stats_ci else None
    agg = run_baseline(settings, extractor, parse_fn, vertical=vertical, collect_rows=collect_rows)
    print(format_report(agg))
    if stats_ci:
        from contract_rag.eval.stats import bootstrap_metric_ci, field_f1_of, source_accuracy_of

        f1_ci = bootstrap_metric_ci(collect_rows, lambda rs: field_f1_of(rs, vertical))
        print(
            f"field_f1 95% CI: [{f1_ci['lo']:.3f}, {f1_ci['hi']:.3f}] "
            f"(bootstrap n={f1_ci['n_boot']}, seed=0)"
        )
        src_ci = bootstrap_metric_ci(collect_rows, lambda rs: source_accuracy_of(rs, vertical))
        print(
            f"source_accuracy 95% CI: [{src_ci['lo']:.3f}, {src_ci['hi']:.3f}] "
            f"(bootstrap n={src_ci['n_boot']}, seed=0)"
        )


if __name__ == "__main__":
    main()
