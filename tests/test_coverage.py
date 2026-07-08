from __future__ import annotations

import pytest

from contract_rag.clean.coverage import (
    DocCoverage,
    PageCoverage,
    annotate_report,
    document_coverage,
    ir_page_boxes,
    ir_page_numbers,
    rollup,
)
from contract_rag.clean.quality import QualityReport, compute_quality_score
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _block(bid, page, x0, y0, x1, y1, engine="paddleocr"):
    return DocBlock(
        block_id=bid, type=BlockType.PARAGRAPH, text="x",
        bbox=BoundingBox(page=page, x0=x0, y0=y0, x1=x1, y1=y1),
        confidence=0.9, source_engine=engine,
    )


def _ir(blocks):
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", blocks=blocks, metadata={})


# ------------------------------------------------------------ pure, numpy-free


def test_ir_page_boxes_scales_from_box_dpi_to_render_dpi():
    ir = _ir([_block("a", 1, 100, 200, 300, 400)])
    # identity at 300/300
    assert ir_page_boxes(ir, 1) == [(100, 200, 300, 400)]
    # half dpi halves coordinates
    assert ir_page_boxes(ir, 1, box_dpi=300, render_dpi=150) == [(50, 100, 150, 200)]


def test_ir_page_boxes_filters_by_page_and_skips_bboxless():
    b_no_bbox = DocBlock(block_id="n", type=BlockType.PARAGRAPH, text="x",
                         confidence=0.9, source_engine="docling")
    ir = _ir([_block("a", 1, 0, 0, 1, 1), _block("b", 2, 5, 5, 6, 6), b_no_bbox])
    assert ir_page_boxes(ir, 1) == [(0, 0, 1, 1)]
    assert ir_page_boxes(ir, 2) == [(5, 5, 6, 6)]
    assert ir_page_numbers(ir) == [1, 2]


def test_rollup_is_ink_weighted():
    # page A: 100 ink, 100 covered (1.0); page B: 100 ink, 0 covered (0.0)
    a = PageCoverage(page=1, height=10, width=10, n_ink=100, n_covered=100,
                     ink_coverage=1.0, uncovered_ink_ratio=0.0, threshold=128)
    b = PageCoverage(page=2, height=10, width=10, n_ink=100, n_covered=0,
                     ink_coverage=0.0, uncovered_ink_ratio=1.0, threshold=128)
    doc = rollup([a, b])
    assert doc.ink_coverage == 0.5  # (100+0)/(100+100)
    assert doc.min_page_ink_coverage == 0.0
    assert doc.max_page_uncovered_ink_ratio == 1.0
    assert doc.total_ink == 200


def test_rollup_empty_raises():
    with pytest.raises(ValueError):
        rollup([])


def test_annotate_report_is_additive_quality_score_untouched():
    report = compute_quality_score(_ir([_block("a", 1, 0, 0, 1, 1)]))
    _cov_keys = ("ink_coverage", "uncovered_ink_ratio")
    before = {k: v for k, v in report.model_dump().items() if k not in _cov_keys}
    doc = DocCoverage(n_pages=1, total_ink=10, ink_coverage=0.8,
                      uncovered_ink_ratio=0.2, min_page_ink_coverage=0.8,
                      max_page_uncovered_ink_ratio=0.2, pages=[])
    annotated = annotate_report(report, doc)
    assert annotated.ink_coverage == 0.8
    assert annotated.uncovered_ink_ratio == 0.2
    # everything else byte-identical
    assert annotated.quality_score == before["quality_score"]
    assert {k: v for k, v in annotated.model_dump().items()
            if k not in _cov_keys} == before
    # original report untouched (immutability)
    assert report.ink_coverage is None


def test_compute_quality_score_leaves_coverage_none_by_default():
    r = compute_quality_score(_ir([_block("a", 1, 0, 0, 1, 1)]))
    assert r.ink_coverage is None
    assert r.uncovered_ink_ratio is None


# ------------------------------------------------------------ numpy-backed core

np = pytest.importorskip("numpy")


def _white(h, w):
    return np.full((h, w), 255, dtype=np.uint8)


def test_ink_mask_otsu_splits_bimodal():
    from contract_rag.clean.coverage import ink_mask, otsu_threshold

    arr = np.array([[0, 0, 255, 255], [0, 0, 255, 255]], dtype=np.uint8)
    thr = otsu_threshold(arr)
    assert 0 <= thr <= 255
    mask = ink_mask(arr)
    # the dark half is ink, the white half is not
    assert mask[:, :2].all()
    assert not mask[:, 2:].any()


def test_page_coverage_full_when_ink_inside_box():
    gray = _white(100, 100)
    gray[40:60, 40:60] = 0  # a 20x20 ink block
    boxes = [(40, 40, 60, 60)]
    pc = page_coverage_call(gray, boxes, dilate_px=0.0)
    assert pc.n_ink == 400
    assert pc.ink_coverage == 1.0
    assert pc.uncovered_ink_ratio == 0.0


def test_page_coverage_detects_uncovered_ink():
    gray = _white(100, 100)
    gray[10:30, 10:30] = 0   # 400 px covered by a box
    gray[70:90, 70:90] = 0   # 400 px with NO box → omission proxy
    boxes = [(10, 10, 30, 30)]
    pc = page_coverage_call(gray, boxes, dilate_px=0.0)
    assert pc.n_ink == 800
    assert pc.ink_coverage == 0.5
    assert pc.uncovered_ink_ratio == 0.5


def test_page_coverage_blank_page_is_fully_covered():
    pc = page_coverage_call(_white(50, 50), [], dilate_px=0.0)
    assert pc.n_ink == 0
    assert pc.ink_coverage == 1.0


def test_dilation_grows_coverage_area():
    from contract_rag.clean.coverage import boxes_mask

    tight = boxes_mask((100, 100), [(40, 40, 60, 60)], dilate_px=0.0)
    grown = boxes_mask((100, 100), [(40, 40, 60, 60)], dilate_px=5.0)
    assert grown.sum() > tight.sum()
    # dilating a box just outside an ink pixel now covers it
    gray = _white(100, 100)
    gray[38:40, 40:42] = 0  # ink 2px above the box top edge
    boxes = [(40, 40, 60, 60)]
    assert page_coverage_call(gray, boxes, dilate_px=0.0).uncovered_ink_ratio > 0
    assert page_coverage_call(gray, boxes, dilate_px=5.0).uncovered_ink_ratio == 0


def test_border_ignore_blanks_edge_ink():
    from contract_rag.clean.coverage import ink_mask

    gray = _white(100, 100)
    gray[0:5, :] = 0  # a black frame along the top edge
    full = ink_mask(gray, border_ignore_frac=0.0)
    trimmed = ink_mask(gray, border_ignore_frac=0.1)
    assert full[0:5, :].any()
    assert not trimmed[0:5, :].any()


def test_document_coverage_uses_render_seam_and_is_deterministic():
    gray1 = _white(100, 100)
    gray1[10:30, 10:30] = 0   # covered
    gray2 = _white(100, 100)
    gray2[70:90, 70:90] = 0   # uncovered
    pages = {0: gray1, 1: gray2}
    ir = _ir([_block("a", 1, 10, 10, 30, 30), _block("b", 2, 0, 0, 1, 1)])

    def render_fn(idx):
        return pages[idx]

    doc = document_coverage(ir, render_fn, dilate_px=0.0)
    doc2 = document_coverage(ir, render_fn, dilate_px=0.0)
    assert doc.model_dump() == doc2.model_dump()  # deterministic
    assert doc.n_pages == 2
    # page 1 fully covered, page 2 entirely uncovered ink
    p1, p2 = doc.pages
    assert p1.ink_coverage == 1.0
    assert p2.ink_coverage == 0.0


def page_coverage_call(gray, boxes, **kw):
    from contract_rag.clean.coverage import page_coverage

    return page_coverage(gray, boxes, **kw)
