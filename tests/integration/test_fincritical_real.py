"""Gated on FINCRITICAL_DIR pointing at a real FinCriticalED snapshot
(raw_input.csv + gold_annotation_html/). Validates the schema mapping against
real records — no OCR run here (that's the `python -m` harness)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

_dir = os.environ.get("FINCRITICAL_DIR", "")
pytestmark = pytest.mark.skipif(
    not (_dir and (Path(_dir) / "raw_input.csv").exists()),
    reason="FINCRITICAL_DIR not set to a FinCriticalED snapshot",
)


def test_load_samples_real_schema():
    from contract_rag.eval.fincritical import load_samples

    samples = load_samples(Path(_dir), cap=5)
    assert samples, "expected at least one sample with a gold file"
    assert [s.page_id for s in samples] == sorted(s.page_id for s in samples)
    for s in samples:
        assert s.image_b64.strip()
        assert s.gold_html.strip()


def test_real_gold_pages_carry_tagged_facts():
    from contract_rag.eval.fincritical import FACT_KINDS, load_samples, parse_gold_html

    samples = load_samples(Path(_dir), cap=10)
    facts = [f for s in samples for f in parse_gold_html(s.gold_html)]
    assert facts, "no fact tags parsed from real gold files — schema drift?"
    assert {f.kind for f in facts} <= set(FACT_KINDS)
    assert all(f.value.strip() for f in facts)


def test_sample_image_decodes_to_a_pdf(tmp_path: Path):
    pytest.importorskip("PIL")
    import pypdfium2 as pdfium

    from contract_rag.eval.fincritical import load_samples, sample_to_pdf

    (sample,) = load_samples(Path(_dir), cap=1)
    pdf = sample_to_pdf(sample, tmp_path)
    doc = pdfium.PdfDocument(str(pdf))
    try:
        assert len(doc) == 1
    finally:
        doc.close()
