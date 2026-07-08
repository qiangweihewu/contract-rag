from __future__ import annotations

from contract_rag.baseline import run_baseline
from contract_rag.clean.pipeline import clean_ir
from contract_rag.clean.quality import compute_quality_score
from contract_rag.config import Settings, assert_backend_allowed, get_settings
from contract_rag.eval.dirtify import dirtify
from contract_rag.eval.golden import load_golden_set


def measure_cleaning_lift(settings: Settings, extractor, parse_fn=None, seed: int = 0) -> dict:
    if parse_fn is None:
        from contract_rag.parse.docling_parser import parse_with_docling

        parse_fn = parse_with_docling

    dirty_agg = run_baseline(settings, extractor, lambda p: dirtify(parse_fn(p), seed=seed))
    cleaned_agg = run_baseline(settings, extractor, lambda p: clean_ir(dirtify(parse_fn(p), seed=seed)))

    quality_pairs = []
    for g in load_golden_set(settings.golden_set_dir):
        base = parse_fn(settings.data_dir / g.source_pdf)
        dirty = dirtify(base, seed=seed)
        quality_pairs.append(
            (compute_quality_score(dirty).quality_score,
             compute_quality_score(clean_ir(dirty)).quality_score)
        )
    return {"dirty_f1": dirty_agg, "cleaned_f1": cleaned_agg, "quality_pairs": quality_pairs}


def format_cleaning_lift(result: dict) -> str:
    d, c, q = result["dirty_f1"], result["cleaned_f1"], result["quality_pairs"]
    mean_dq = sum(x for x, _ in q) / len(q) if q else 0.0
    mean_cq = sum(y for _, y in q) / len(q) if q else 0.0
    return "\n".join([
        "=== Cleaning lift (dirty -> cleaned) ===",
        f"docs:          {d['n_docs']}",
        f"field_f1:      dirty={d['field_f1']:.3f}  cleaned={c['field_f1']:.3f}  "
        f"delta={c['field_f1'] - d['field_f1']:+.3f}",
        f"quality_score: dirty={mean_dq:.3f}  cleaned={mean_cq:.3f}  delta={mean_cq - mean_dq:+.3f}",
    ])


def main() -> None:
    import os
    from pathlib import Path

    from contract_rag.eval.ir_cache import ir_cache
    from contract_rag.extract.extractor import get_extractor
    from contract_rag.parse.docling_parser import parse_with_docling

    settings = get_settings()
    assert_backend_allowed(settings)
    extractor = get_extractor(settings)
    parse_fn = ir_cache(Path(os.environ.get("IR_CACHE_DIR", ".ir_cache")), parse_with_docling)
    print(format_cleaning_lift(measure_cleaning_lift(settings, extractor, parse_fn=parse_fn)))


if __name__ == "__main__":
    main()
