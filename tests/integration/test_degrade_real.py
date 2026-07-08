"""Gated end-to-end image degradation: render a real CUAD digital page → degrade →
paddleocr → score. Skips unless a golden set + data/ exist and paddleocr/pypdfium2 are
installed (the actual quality-drop measurement lives in the `python -m` harness; this
guards the render→degrade→OCR→score seam on one real doc, capped to 1 page for speed)."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PIL")
pytest.importorskip("numpy")
pytest.importorskip("pypdfium2")
pytest.importorskip("paddleocr")

from contract_rag.config import get_settings
from contract_rag.eval.degrade import degrade_pdf
from contract_rag.eval.golden import load_golden_set
from contract_rag.clean.quality import compute_quality_score
from contract_rag.clean.pipeline import clean_ir
from contract_rag.parse.paddle_parser import parse_with_paddle
from contract_rag.parse.docling_parser import parse_with_docling


def _first_source_pdf(settings) -> Path | None:
    golden = load_golden_set(settings.golden_set_dir)
    for g in golden:
        p = settings.data_dir / g.source_pdf
        if p.exists():
            return p
    return None


settings = get_settings()
_src = _first_source_pdf(settings) if settings.golden_set_dir.exists() else None
pytestmark = pytest.mark.skipif(
    _src is None, reason="no golden set + data/ PDFs present"
)


def test_degradation_drops_quality_below_clean(tmp_path: Path):
    # original (clean digital) parse — high quality baseline
    original = compute_quality_score(parse_with_docling(_src))

    out_pdf = tmp_path / "degraded.pdf"
    degrade_pdf(_src, out_pdf, seed=0, level="fax", max_pages=1)
    degraded_ir = parse_with_paddle(out_pdf)
    degraded = compute_quality_score(degraded_ir)
    cleaned = compute_quality_score(clean_ir(degraded_ir))

    assert degraded_ir.blocks, "OCR produced no blocks from the degraded page"
    assert all(b.source_engine == "paddleocr" for b in degraded_ir.blocks)
    # the whole point: image-level degradation drops quality below the clean digital parse
    assert degraded.quality_score < original.quality_score
    # cleaning never makes it worse
    assert cleaned.quality_score >= degraded.quality_score
