"""Layout-model coverage — the finer-grained sibling of `clean/coverage.py`.

`clean/coverage.py`'s geometric ink-coverage signal is a strong region-scale
omission detector (Tobacco800 GEDI signature/logo zones: 5.4x more uncovered ink
inside the zone than outside) but a weak single-fact detector (FinCriticalED
point-biserial 0.18, pearson 0.15) — CLAUDE.md's honest conclusion is that a
dropped financial *number* is a few pixels among thousands, so it barely dents a
whole-page ink ratio even when OCR silently drops it.

This module tests the fix CLAUDE.md names: instead of asking "is the ink on this
*page* accounted for", ask "is each region a LAYOUT DETECTOR finds actually filled
by an OCR block". A layout model (PaddleOCR's `LayoutDetection`, PP-DocLayout_plus-L)
segments a page into semantic regions (table/text/title/figure/...) independent of
whether OCR produced any text for them. A region with little or no OCR-block
coverage — a table the OCR skipped, a paragraph that silently vanished — is a
*region-scale* omission candidate, scored at a much finer granularity than one
number per page.

Structure mirrors `clean/coverage.py` exactly: a pure core (regions + OCR-block
boxes -> fill ratios -> uncovered regions -> a doc-level omission score) that never
touches numpy/paddleocr directly beyond `boxes_mask` rasterization, with the model
call and page render behind injectable seams (`regions_fn` on
`document_layout_coverage`, mirroring `coverage.document_coverage`'s `render_fn`) so
unit tests use hand-built regions and never load paddleocr. Integration into
`QualityReport` is additive-only, exactly like `coverage.annotate_report`:
`compute_quality_score`'s existing fields are byte-for-byte unchanged (see the
regression test in `tests/test_layout_coverage.py`); `layout_omission_score` /
`n_uncovered_regions` are new optional fields, populated only via
`annotate_report_layout` on the scanned/paddle path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from pydantic import BaseModel, Field

from contract_rag.clean.coverage import boxes_mask, ir_page_boxes, ir_page_numbers
from contract_rag.clean.quality import QualityReport
from contract_rag.ir import DocumentIR

_PADDLE_RENDER_DPI = 300.0  # matches clean/coverage.py + paddle_parser's box space
_DEFAULT_FILL_THRESHOLD = 0.5  # region counted "covered" if >=50% of its area sits
                                # inside some (dilated) OCR block
Box = tuple[float, float, float, float]  # (x0, y0, x1, y1), top-left origin


# ============================================================ pure region-fill core


class LayoutRegion(BaseModel):
    """One layout-detector region on a page, rendered-px convention (same as
    `clean/coverage.py`'s `Box`)."""

    label: str
    score: float
    box: Box


class RegionFill(BaseModel):
    """One region's OCR-block coverage. `fill_ratio` is the fraction of the
    region's own area that falls inside some (dilated) OCR block; `covered` is
    the threshold verdict used to build `uncovered_regions`."""

    label: str
    score: float
    box: Box
    area: int
    fill_ratio: float
    covered: bool


def region_fill_ratio(
    region: Box, ocr_boxes: Sequence[Box], *, dilate_px: float = 0.0
) -> tuple[float, int]:
    """Pure: fraction of `region`'s area covered by any `ocr_boxes` box (each
    dilated by `dilate_px`, the same glyph-edge tolerance `clean.coverage` uses).
    Rasterizes onto a canvas local to the region — never a full page-sized array —
    so this scales to per-region calls with no page-shape dependency. A
    degenerate (zero-area) region is fully covered by convention (nothing to
    miss); its reported area is 0."""
    x0, x1 = sorted((region[0], region[2]))
    y0, y1 = sorted((region[1], region[3]))
    w = int(round(x1 - x0))
    h = int(round(y1 - y0))
    if w <= 0 or h <= 0:
        return 1.0, 0

    local_boxes = [(bx0 - x0, by0 - y0, bx1 - x0, by1 - y0) for bx0, by0, bx1, by1 in ocr_boxes]
    covered = boxes_mask((h, w), local_boxes, dilate_px=dilate_px)
    return float(covered.mean()), h * w


class PageLayoutCoverage(BaseModel):
    """One page's layout-region coverage. `layout_omission_score` is the
    area-weighted fraction of detected-region area left uncovered — a table the
    OCR dropped entirely counts more than a one-line caption."""

    page: int
    n_regions: int
    n_uncovered: int
    layout_omission_score: float
    threshold: float
    regions: list[RegionFill] = Field(default_factory=list)
    uncovered_regions: list[RegionFill] = Field(default_factory=list)


def page_layout_coverage(
    regions: Sequence[LayoutRegion],
    ocr_boxes: Sequence[Box],
    *,
    page: int = 1,
    fill_threshold: float = _DEFAULT_FILL_THRESHOLD,
    dilate_px: float = 4.0,
) -> PageLayoutCoverage:
    """Pure per-page rollup. A page with no detected regions is reported with
    `layout_omission_score=0.0` by convention (the layout model found nothing to
    check, not "everything is missing")."""
    filled: list[RegionFill] = []
    for r in regions:
        ratio, area = region_fill_ratio(r.box, ocr_boxes, dilate_px=dilate_px)
        filled.append(RegionFill(
            label=r.label, score=r.score, box=r.box, area=area,
            fill_ratio=round(ratio, 4), covered=ratio >= fill_threshold,
        ))
    uncovered = [f for f in filled if not f.covered]
    total_area = sum(f.area for f in filled)
    uncovered_area = sum(f.area for f in uncovered)
    omission = (uncovered_area / total_area) if total_area else 0.0
    return PageLayoutCoverage(
        page=page,
        n_regions=len(filled),
        n_uncovered=len(uncovered),
        layout_omission_score=round(omission, 4),
        threshold=fill_threshold,
        regions=filled,
        uncovered_regions=uncovered,
    )


class DocLayoutCoverage(BaseModel):
    """Document-level roll-up, area-weighted like `PageLayoutCoverage` (a big
    uncovered table counts more than a small one), plus the single worst page for
    HITL routing."""

    n_pages: int
    n_regions: int
    n_uncovered: int
    layout_omission_score: float
    mean_page_omission: float
    max_page_omission: float
    pages: list[PageLayoutCoverage] = Field(default_factory=list)


def _page_area(pc: PageLayoutCoverage) -> tuple[int, int]:
    total = sum(f.area for f in pc.regions)
    uncov = sum(f.area for f in pc.uncovered_regions)
    return total, uncov


def rollup(pages: Sequence[PageLayoutCoverage]) -> DocLayoutCoverage:
    if not pages:
        raise ValueError("rollup of no pages")
    total_area = uncov_area = 0
    for p in pages:
        t, u = _page_area(p)
        total_area += t
        uncov_area += u
    omission = (uncov_area / total_area) if total_area else 0.0
    return DocLayoutCoverage(
        n_pages=len(pages),
        n_regions=sum(p.n_regions for p in pages),
        n_uncovered=sum(p.n_uncovered for p in pages),
        layout_omission_score=round(omission, 4),
        mean_page_omission=round(sum(p.layout_omission_score for p in pages) / len(pages), 4),
        max_page_omission=round(max(p.layout_omission_score for p in pages), 4),
        pages=list(pages),
    )


# ============================================================ IR + doc-level wiring


def document_layout_coverage(
    ir: DocumentIR,
    regions_fn: Callable[[int], Sequence[LayoutRegion]],
    *,
    box_dpi: float = _PADDLE_RENDER_DPI,
    render_dpi: float = _PADDLE_RENDER_DPI,
    fill_threshold: float = _DEFAULT_FILL_THRESHOLD,
    dilate_px: float = 4.0,
) -> DocLayoutCoverage:
    """Layout coverage for every page carrying a bbox. `regions_fn(page_index_0based)`
    returns that page's detected `LayoutRegion`s — the injectable seam: a real
    layout-model call (`detect_page_regions_from_pdf`), or a fake in tests. OCR
    boxes come from the IR itself (`ir_page_boxes`, same convention as
    `clean.coverage`), scaled from `box_dpi` to `render_dpi` so regions and OCR
    boxes share a pixel space."""
    pages = ir_page_numbers(ir)
    if not pages:
        raise ValueError("IR has no bbox-bearing pages; layout coverage is undefined")
    out: list[PageLayoutCoverage] = []
    for pg in pages:
        regions = regions_fn(pg - 1)
        boxes = ir_page_boxes(ir, pg, box_dpi=box_dpi, render_dpi=render_dpi)
        out.append(page_layout_coverage(
            regions, boxes, page=pg, fill_threshold=fill_threshold, dilate_px=dilate_px,
        ))
    return rollup(out)


# ============================================================ layout-model seam (impure)


_LAYOUT_SINGLETON: tuple | None = None  # (model, api) — model load is seconds, reuse it


def _select_layout_api(version: str) -> str:
    """Pick the layout-detection call surface from `paddleocr.__version__`:
    `LayoutDetection` (PP-Structure v3 family) for >=3, the legacy `PPStructure`
    layout-only mode for 2.x. Pure/testable without the dependency installed,
    mirroring `paddle_parser._select_api`."""
    try:
        major = int(version.strip().split(".")[0])
    except (ValueError, IndexError):
        raise ValueError(f"unparseable paddleocr version {version!r}") from None
    return "predict" if major >= 3 else "legacy"


def _get_layout_model() -> tuple:
    global _LAYOUT_SINGLETON
    if _LAYOUT_SINGLETON is None:
        import paddleocr

        api = _select_layout_api(paddleocr.__version__)
        try:
            if api == "predict":  # paddleocr >= 3
                model = paddleocr.LayoutDetection()
            else:  # paddleocr 2.x: PPStructure's layout-only mode
                model = paddleocr.PPStructure(layout=True, table=False, ocr=False, show_log=False)
        except Exception as exc:
            raise RuntimeError(
                f"failed to construct layout-detection model (detected api={api!r} for "
                f"paddleocr {paddleocr.__version__}): {exc}"
            ) from exc
        _LAYOUT_SINGLETON = (model, api)
    return _LAYOUT_SINGLETON


def predict_result_to_regions(res) -> list[LayoutRegion]:
    """paddleocr >= 3 `LayoutDetection.predict()` result (dict-like with a
    `boxes` list of `{"label", "score", "coordinate": [x0, y0, x1, y1]}`) ->
    `LayoutRegion` list. Pure: takes any mapping, so unit tests pass a plain dict
    (mirrors `paddle_parser.predict_result_to_lines`)."""
    out: list[LayoutRegion] = []
    for b in res["boxes"]:
        x0, y0, x1, y1 = (float(v) for v in b["coordinate"])
        out.append(LayoutRegion(
            label=str(b.get("label", "region")),
            score=float(b.get("score", 0.0)),
            box=(x0, y0, x1, y1),
        ))
    return out


def legacy_result_to_regions(result_page) -> list[LayoutRegion]:
    """paddleocr 2.x `PPStructure(...)(image)` result: a list of
    `{"type": ..., "bbox": [x0, y0, x1, y1], ...}` dicts -> `LayoutRegion` list.
    The legacy API doesn't emit a per-region confidence, so `score` defaults to
    1.0 (mirrors `paddle_parser.legacy_result_to_lines`'s tolerance)."""
    out: list[LayoutRegion] = []
    for item in result_page or []:
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = (float(v) for v in bbox)
        out.append(LayoutRegion(label=str(item.get("type", "region")), score=1.0, box=(x0, y0, x1, y1)))
    return out


def detect_regions(image_path: Path) -> list[LayoutRegion]:
    """Run the (lazily constructed, cached) layout-detection model on a single
    page image and normalize its result to `LayoutRegion`s."""
    model, api = _get_layout_model()
    if api == "predict":
        regions: list[LayoutRegion] = []
        for res in model.predict(str(image_path)):
            regions.extend(predict_result_to_regions(res))
        return regions
    result = model(str(image_path))
    return legacy_result_to_regions(result)


def detect_page_regions_from_pdf(
    pdf_path: Path, page_index: int, dpi: float = _PADDLE_RENDER_DPI
) -> list[LayoutRegion]:
    """Render 0-based `page_index` of `pdf_path` to a temp PNG at `dpi` (the
    paddle box-space default) and run layout detection on it. Lazy
    pypdfium2/paddleocr (already runtime deps). This is the real `regions_fn` to
    hand `document_layout_coverage`; tests inject a fake instead."""
    import tempfile

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_index]
        try:
            pil = page.render(scale=dpi / 72).to_pil()
        finally:
            page.close()
    finally:
        pdf.close()
    with tempfile.TemporaryDirectory() as d:
        img_path = Path(d) / "page.png"
        pil.save(img_path)
        return detect_regions(img_path)


# ============================================================ additive integration


def annotate_report_layout(report: QualityReport, coverage: DocLayoutCoverage) -> QualityReport:
    """Return a copy of `report` with the two layout-coverage fields populated.
    Additive only, exactly like `coverage.annotate_report`: `quality_score` and
    every other existing field are untouched — this is a *new* signal alongside
    the formula (and alongside the geometric ink-coverage signal, which uses its
    own two fields), never folded in by default."""
    return report.model_copy(update={
        "layout_omission_score": coverage.layout_omission_score,
        "n_uncovered_regions": coverage.n_uncovered,
    })
