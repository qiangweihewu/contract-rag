from __future__ import annotations

from contract_rag.baseline import run_baseline
from contract_rag.config import Settings, assert_backend_allowed, get_settings
from contract_rag.parse.docling_parser import parse_with_docling


def compare_parsers(
    settings: Settings, extractor, docling_fn=parse_with_docling, router_fn=None,
    *, collect_docling_rows: list | None = None, collect_router_rows: list | None = None,
):
    if router_fn is None:
        from contract_rag.parse.router import parse

        router_fn = lambda p: parse(p, settings)  # noqa: E731
    docling_agg = run_baseline(settings, extractor, docling_fn, collect_rows=collect_docling_rows)
    router_agg = run_baseline(settings, extractor, router_fn, collect_rows=collect_router_rows)
    return docling_agg, router_agg


def format_comparison(docling_agg: dict, router_agg: dict, stats: dict | None = None) -> str:
    d = docling_agg
    r = router_agg
    lines = [
        "=== Parser comparison (docling baseline vs router) ===",
        f"docs:            {r['n_docs']}",
        f"field_f1:        docling={d['field_f1']:.3f}  router={r['field_f1']:.3f}  "
        f"delta={r['field_f1'] - d['field_f1']:+.3f}",
        f"source_accuracy: docling={d['source_accuracy']:.3f}  router={r['source_accuracy']:.3f}  "
        f"delta={r['source_accuracy'] - d['source_accuracy']:+.3f}",
        "per-field accuracy (docling -> router):",
    ]
    for name in r["per_field"]:
        lines.append(f"  {name:<18} {d['per_field'][name]:.3f} -> {r['per_field'][name]:.3f}")
    if stats is not None:
        lines.append(
            f"p-value (paired permutation, field_f1): {stats['p_value']:.4f} "
            f"(n_perm={stats['n_perm']})"
        )
    return "\n".join(lines)


def main() -> None:
    import os

    from contract_rag.extract.extractor import get_extractor

    settings = get_settings()
    assert_backend_allowed(settings)
    extractor = get_extractor(settings)
    stats_ci = bool(os.environ.get("STATS_CI"))
    if stats_ci:
        docling_rows: list = []
        router_rows: list = []
        docling_agg, router_agg = compare_parsers(
            settings, extractor,
            collect_docling_rows=docling_rows, collect_router_rows=router_rows,
        )
        from contract_rag.eval.stats import field_f1_of, paired_permutation_test
        from contract_rag.verticals.registry import get_vertical_for

        # Bind the resolved vertical: the rows are shaped by settings' vertical, so the
        # default-vertical fallback would iterate the wrong field_names (KeyError under
        # e.g. VERTICAL=nda). Mirrors baseline.main()'s closure.
        vertical = get_vertical_for(settings)
        stats = paired_permutation_test(
            docling_rows, router_rows, lambda rs: field_f1_of(rs, vertical)
        )
        print(format_comparison(docling_agg, router_agg, stats=stats))
    else:
        docling_agg, router_agg = compare_parsers(settings, extractor)
        print(format_comparison(docling_agg, router_agg))


if __name__ == "__main__":
    main()
