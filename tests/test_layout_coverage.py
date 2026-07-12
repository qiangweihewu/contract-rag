from __future__ import annotations

import pytest

from contract_rag.clean.layout_coverage import (
    DocLayoutCoverage,
    LayoutRegion,
    PageLayoutCoverage,
    RegionFill,
    _select_layout_api,
    annotate_report_layout,
    document_layout_coverage,
    legacy_result_to_regions,
    page_layout_coverage,
    predict_result_to_regions,
    region_fill_ratio,
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


def _region(label, x0, y0, x1, y1, score=0.9):
    return LayoutRegion(label=label, score=score, box=(x0, y0, x1, y1))


# ------------------------------------------------------------ numpy-backed pure core

np = pytest.importorskip("numpy")


def test_region_fill_ratio_full_when_ocr_box_covers_region():
    ratio, area = region_fill_ratio((0, 0, 100, 100), [(0, 0, 100, 100)], dilate_px=0.0)
    assert ratio == 1.0
    assert area == 10000


def test_region_fill_ratio_partial_coverage():
    # OCR box covers the left half of a 100x100 region only
    ratio, area = region_fill_ratio((0, 0, 100, 100), [(0, 0, 50, 100)], dilate_px=0.0)
    assert ratio == pytest.approx(0.5, abs=1e-3)
    assert area == 10000


def test_region_fill_ratio_no_ocr_boxes_is_zero():
    ratio, area = region_fill_ratio((0, 0, 100, 100), [], dilate_px=0.0)
    assert ratio == 0.0
    assert area == 10000


def test_region_fill_ratio_zero_area_region_is_fully_covered_by_convention():
    ratio, area = region_fill_ratio((10, 10, 10, 50), [], dilate_px=0.0)
    assert ratio == 1.0
    assert area == 0


def test_region_fill_ratio_ignores_ocr_boxes_outside_region():
    # OCR box entirely outside the region contributes nothing
    ratio, area = region_fill_ratio((0, 0, 100, 100), [(200, 200, 300, 300)], dilate_px=0.0)
    assert ratio == 0.0


def test_region_fill_ratio_dilation_grows_coverage():
    # OCR box just outside the region; dilation should pull it in
    tight, _ = region_fill_ratio((0, 0, 100, 100), [(100, 0, 102, 100)], dilate_px=0.0)
    grown, _ = region_fill_ratio((0, 0, 100, 100), [(100, 0, 102, 100)], dilate_px=5.0)
    assert tight == 0.0
    assert grown > 0.0


# ------------------------------------------------------------ page/doc rollup


def test_page_layout_coverage_threshold_behavior():
    regions = [
        _region("table", 0, 0, 100, 100),   # fully covered
        _region("text", 0, 200, 100, 300),  # fully uncovered
    ]
    ocr_boxes = [(0, 0, 100, 100)]
    pc = page_layout_coverage(regions, ocr_boxes, page=1, fill_threshold=0.5, dilate_px=0.0)
    assert pc.n_regions == 2
    assert pc.n_uncovered == 1
    assert pc.uncovered_regions[0].label == "text"
    # area-weighted: both regions are 10000px, half uncovered -> 0.5
    assert pc.layout_omission_score == 0.5


def test_page_layout_coverage_no_regions_is_zero_omission():
    pc = page_layout_coverage([], [(0, 0, 10, 10)], page=1)
    assert pc.n_regions == 0
    assert pc.layout_omission_score == 0.0


def test_page_layout_coverage_threshold_is_configurable():
    regions = [_region("text", 0, 0, 100, 100)]
    ocr_boxes = [(0, 0, 100, 60)]  # 60% fill
    strict = page_layout_coverage(regions, ocr_boxes, fill_threshold=0.9, dilate_px=0.0)
    lenient = page_layout_coverage(regions, ocr_boxes, fill_threshold=0.5, dilate_px=0.0)
    assert strict.n_uncovered == 1
    assert lenient.n_uncovered == 0


def test_rollup_is_area_weighted():
    # page A: one 100x100 region fully covered; page B: one 100x100 region fully uncovered
    a = page_layout_coverage([_region("t", 0, 0, 100, 100)], [(0, 0, 100, 100)], page=1, dilate_px=0.0)
    b = page_layout_coverage([_region("t", 0, 0, 100, 100)], [], page=2, dilate_px=0.0)
    doc = rollup([a, b])
    assert doc.n_pages == 2
    assert doc.n_regions == 2
    assert doc.n_uncovered == 1
    assert doc.layout_omission_score == 0.5
    assert doc.max_page_omission == 1.0


def test_rollup_empty_raises():
    with pytest.raises(ValueError):
        rollup([])


def test_document_layout_coverage_uses_regions_seam():
    ir = _ir([_block("a", 1, 0, 0, 100, 100), _block("b", 2, 0, 0, 1, 1)])
    fake_regions = {
        0: [_region("table", 0, 0, 100, 100)],   # page 1 (0-based idx 0): covered
        1: [_region("text", 50, 50, 150, 150)],  # page 2 (0-based idx 1): uncovered (block is tiny)
    }

    def regions_fn(idx):
        return fake_regions[idx]

    doc = document_layout_coverage(ir, regions_fn, dilate_px=0.0)
    assert doc.n_pages == 2
    p1, p2 = doc.pages
    assert p1.layout_omission_score == 0.0
    assert p2.layout_omission_score == 1.0


def test_document_layout_coverage_no_bbox_pages_raises():
    ir = _ir([DocBlock(block_id="n", type=BlockType.PARAGRAPH, text="x",
                       confidence=0.9, source_engine="docling")])
    with pytest.raises(ValueError):
        document_layout_coverage(ir, lambda i: [])


def test_document_layout_coverage_is_deterministic():
    ir = _ir([_block("a", 1, 0, 0, 100, 100)])
    regions_fn = lambda idx: [_region("table", 0, 0, 100, 100)]
    d1 = document_layout_coverage(ir, regions_fn, dilate_px=0.0)
    d2 = document_layout_coverage(ir, regions_fn, dilate_px=0.0)
    assert d1.model_dump() == d2.model_dump()


# ------------------------------------------------------------ layout-model result normalization (dep-free)


def test_predict_result_to_regions_normalizes_v3_dict():
    res = {
        "boxes": [
            {"cls_id": 8, "label": "table", "score": 0.96, "coordinate": [0.0, 10.0, 100.0, 200.0]},
            {"cls_id": 2, "label": "text", "score": 0.88, "coordinate": [5.0, 5.0, 50.0, 50.0]},
        ]
    }
    regions = predict_result_to_regions(res)
    assert regions == [
        LayoutRegion(label="table", score=0.96, box=(0.0, 10.0, 100.0, 200.0)),
        LayoutRegion(label="text", score=0.88, box=(5.0, 5.0, 50.0, 50.0)),
    ]


def test_legacy_result_to_regions_normalizes_v2_list():
    page = [
        {"type": "table", "bbox": [0, 10, 100, 200]},
        {"type": "text", "bbox": [5, 5, 50, 50]},
        {"type": "figure"},  # missing bbox -> skipped
    ]
    regions = legacy_result_to_regions(page)
    assert len(regions) == 2
    assert regions[0].label == "table"
    assert regions[0].score == 1.0
    assert regions[0].box == (0.0, 10.0, 100.0, 200.0)


def test_legacy_result_to_regions_none_page_is_empty():
    assert legacy_result_to_regions(None) == []


@pytest.mark.parametrize(
    "version,expected",
    [
        ("3.0.0", "predict"),
        ("3.7.0", "predict"),
        ("4.1.2", "predict"),
        ("2.7.0.3", "legacy"),
        ("2.6.1", "legacy"),
    ],
)
def test_select_layout_api_by_major_version(version, expected):
    assert _select_layout_api(version) == expected


def test_select_layout_api_raises_on_unparseable_version():
    with pytest.raises(ValueError, match="unparseable paddleocr version"):
        _select_layout_api("not-a-version")


# ------------------------------------------------------------ additive QualityReport integration


def test_annotate_report_layout_is_additive_quality_score_untouched():
    report = compute_quality_score(_ir([_block("a", 1, 0, 0, 1, 1)]))
    _cov_keys = ("layout_omission_score", "n_uncovered_regions")
    before = {k: v for k, v in report.model_dump().items() if k not in _cov_keys}
    doc = DocLayoutCoverage(
        n_pages=1, n_regions=4, n_uncovered=1, layout_omission_score=0.25,
        mean_page_omission=0.25, max_page_omission=0.25, pages=[],
    )
    annotated = annotate_report_layout(report, doc)
    assert annotated.layout_omission_score == 0.25
    assert annotated.n_uncovered_regions == 1
    # everything else byte-identical, including the (separate) ink-coverage fields
    assert annotated.quality_score == before["quality_score"]
    assert {k: v for k, v in annotated.model_dump().items() if k not in _cov_keys} == before
    # original report untouched (immutability)
    assert report.layout_omission_score is None
    assert report.n_uncovered_regions is None


def test_compute_quality_score_leaves_layout_coverage_none_by_default():
    r = compute_quality_score(_ir([_block("a", 1, 0, 0, 1, 1)]))
    assert r.layout_omission_score is None
    assert r.n_uncovered_regions is None


def test_layout_and_ink_coverage_fields_are_independent():
    # annotating with layout coverage must not touch the geometric ink-coverage
    # fields (and vice versa is already covered by tests/test_coverage.py)
    report = compute_quality_score(_ir([_block("a", 1, 0, 0, 1, 1)]))
    doc = DocLayoutCoverage(
        n_pages=1, n_regions=1, n_uncovered=0, layout_omission_score=0.0,
        mean_page_omission=0.0, max_page_omission=0.0, pages=[],
    )
    annotated = annotate_report_layout(report, doc)
    assert annotated.ink_coverage is None
    assert annotated.uncovered_ink_ratio is None
