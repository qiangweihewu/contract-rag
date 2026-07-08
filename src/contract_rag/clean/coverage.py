"""Geometric *ink-coverage* — an omission-aware quality signal.

Four independent measurement lanes converged on one gap (see CLAUDE.md
Implementation status): the `quality.compute_quality_score` formula scores *what
OCR emitted*, not *what it missed*. A dropped signature, an unread stamp, a
skipped table cell produce **no block**, so garble/empty/confidence never see the
loss — real Tobacco800 scans read at mean quality 0.987 while 7.7% of expert
facts vanish from the OCR output.

This module builds the complementary signal without any new model dependency:
render the page to grayscale, threshold to an "ink" mask (dark pixels = visible
content — text, stamps, signatures, faint regions), and measure the fraction of
ink pixels that fall **inside** some OCR block's bbox versus outside. Low
coverage = OCR failed to account for visible content = likely omission. It reuses
machinery already in the repo: the rendered page (pypdfium2, like `demo.render` /
`eval.degrade`), the block bboxes (paddle convention = rendered px @300dpi
top-left, per `demo.highlight`), and Otsu thresholding (hoisted from
`eval.degrade`).

Everything geometric here is a **pure function of (grayscale array, boxes)** —
render/IO is behind injectable seams, so unit tests use hand-built masks and fake
bboxes where the coverage is known exactly, with no OCR/PDF. It is **additive**:
it never touches `quality_score` (see `annotate_report`); the coverage numbers are
new optional fields, populated only on the scanned/paddle path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from pydantic import BaseModel

from contract_rag.clean.quality import QualityReport
from contract_rag.ir import DocumentIR

_PADDLE_RENDER_DPI = 300.0  # paddle_parser renders (and boxes) at this dpi
Box = tuple[float, float, float, float]  # (x0, y0, x1, y1), top-left origin


# ============================================================ pure ink / boxes core


def otsu_threshold(arr) -> int:
    """Otsu's method: the 0..255 threshold maximizing inter-class variance. Pure
    numpy. Canonical home for the helper `eval.degrade` re-exports."""
    import numpy as np

    hist, _ = np.histogram(arr.ravel(), bins=256, range=(0, 256))
    total = arr.size
    if total == 0:
        return 128
    p = hist.astype("float64") / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 0, (mu_t * omega - mu) ** 2 / denom, 0.0)
    return int(np.argmax(sigma_b))


def ink_mask(gray, threshold: int | None = None, border_ignore_frac: float = 0.0):
    """Boolean mask of "ink" pixels (dark = content). `threshold=None` uses Otsu;
    pixels strictly below the threshold are ink, so a near-white background never
    counts. `border_ignore_frac` blanks a margin of that fraction of width/height
    on each edge — scanner frames / black page borders are ink no OCR box will ever
    cover and would otherwise read as a permanent omission."""
    import numpy as np

    arr = np.asarray(gray)
    if arr.ndim != 2:
        raise ValueError("ink_mask expects a 2-D grayscale array")
    thr = otsu_threshold(arr) if threshold is None else int(threshold)
    # Otsu returns the split index i: class-0 (dark/ink) = values [0..i]. So ink is
    # value <= thr (a pure-black/white page splits at 0 and only value-0 is ink).
    mask = arr <= thr
    if border_ignore_frac > 0:
        h, w = mask.shape
        my, mx = int(round(h * border_ignore_frac)), int(round(w * border_ignore_frac))
        if my or mx:
            keep = np.zeros_like(mask)
            keep[my : h - my, mx : w - mx] = True
            mask = mask & keep
    return mask


def boxes_mask(shape: tuple[int, int], boxes: Sequence[Box], dilate_px: float = 0.0):
    """Rasterize `boxes` (in the mask's own pixel space) into a boolean coverage
    mask of `shape` (h, w). Each box is expanded by `dilate_px` on every side
    before rasterizing — text ink slightly exceeds its tight OCR box, so a few
    pixels of dilation measures genuine gaps rather than glyph-edge anti-aliasing.
    Boxes are clamped to the page; degenerate/out-of-page boxes are skipped."""
    import numpy as np

    h, w = shape
    covered = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        lo_x = max(0, int(np.floor(min(x0, x1) - dilate_px)))
        hi_x = min(w, int(np.ceil(max(x0, x1) + dilate_px)))
        lo_y = max(0, int(np.floor(min(y0, y1) - dilate_px)))
        hi_y = min(h, int(np.ceil(max(y0, y1) + dilate_px)))
        if hi_x > lo_x and hi_y > lo_y:
            covered[lo_y:hi_y, lo_x:hi_x] = True
    return covered


def uncovered_ink_mask(
    gray,
    boxes: Sequence[Box],
    *,
    threshold: int | None = None,
    dilate_px: float = 4.0,
    border_ignore_frac: float = 0.0,
):
    """Boolean mask of ink pixels NOT accounted for by any (dilated) OCR box — the
    omission proxy, spatially localized. Used by the Tobacco800 signature-zone
    validation to ask *where* the uncovered ink lands."""
    mask = ink_mask(gray, threshold, border_ignore_frac)
    covered = boxes_mask(mask.shape, boxes, dilate_px)
    return mask & ~covered


class PageCoverage(BaseModel):
    """One page's ink-coverage measurement. `ink_coverage` is the fraction of ink
    pixels accounted for by some OCR box; `uncovered_ink_ratio` is its complement —
    the omission proxy. A page with no ink is fully covered by convention (nothing
    was missed)."""

    page: int
    height: int
    width: int
    n_ink: int
    n_covered: int
    ink_coverage: float
    uncovered_ink_ratio: float
    threshold: int


def page_coverage(
    gray,
    boxes: Sequence[Box],
    *,
    page: int = 1,
    threshold: int | None = None,
    dilate_px: float = 4.0,
    border_ignore_frac: float = 0.0,
) -> PageCoverage:
    """Pure per-page coverage. `boxes` must already be in `gray`'s pixel space
    (the impure layer scales IR boxes from paddle's 300 dpi to the render dpi)."""
    import numpy as np

    mask = ink_mask(gray, threshold, border_ignore_frac)
    covered = boxes_mask(mask.shape, boxes, dilate_px)
    n_ink = int(mask.sum())
    n_covered = int((mask & covered).sum())
    coverage = 1.0 if n_ink == 0 else n_covered / n_ink
    thr = otsu_threshold(np.asarray(gray)) if threshold is None else int(threshold)
    return PageCoverage(
        page=page,
        height=mask.shape[0],
        width=mask.shape[1],
        n_ink=n_ink,
        n_covered=n_covered,
        ink_coverage=round(coverage, 4),
        uncovered_ink_ratio=round(1.0 - coverage, 4),
        threshold=thr,
    )


class DocCoverage(BaseModel):
    """Document-level roll-up. `ink_coverage` is ink-weighted (total covered ink /
    total ink), so a large omission on a text-dense page counts more than the same
    ratio on a near-blank one; `min_page_ink_coverage` /
    `max_page_uncovered_ink_ratio` surface the single worst page for HITL routing."""

    n_pages: int
    total_ink: int
    ink_coverage: float
    uncovered_ink_ratio: float
    min_page_ink_coverage: float
    max_page_uncovered_ink_ratio: float
    pages: list[PageCoverage]


def rollup(pages: Sequence[PageCoverage]) -> DocCoverage:
    if not pages:
        raise ValueError("rollup of no pages")
    total_ink = sum(p.n_ink for p in pages)
    total_cov = sum(p.n_covered for p in pages)
    coverage = 1.0 if total_ink == 0 else total_cov / total_ink
    return DocCoverage(
        n_pages=len(pages),
        total_ink=total_ink,
        ink_coverage=round(coverage, 4),
        uncovered_ink_ratio=round(1.0 - coverage, 4),
        min_page_ink_coverage=round(min(p.ink_coverage for p in pages), 4),
        max_page_uncovered_ink_ratio=round(max(p.uncovered_ink_ratio for p in pages), 4),
        pages=list(pages),
    )


# ============================================================ IR box extraction


def ir_page_boxes(ir: DocumentIR, page: int, box_dpi: float = _PADDLE_RENDER_DPI,
                  render_dpi: float = _PADDLE_RENDER_DPI) -> list[Box]:
    """Boxes of every block on 1-based `page`, scaled from the parser's box dpi to
    the render dpi (identity when both are 300, the paddle default). Blocks without
    a bbox are skipped."""
    s = render_dpi / box_dpi
    out: list[Box] = []
    for b in ir.blocks:
        bb = b.bbox
        if bb is None or bb.page != page:
            continue
        out.append((bb.x0 * s, bb.y0 * s, bb.x1 * s, bb.y1 * s))
    return out


def ir_page_numbers(ir: DocumentIR) -> list[int]:
    """Sorted distinct 1-based page numbers carrying a bbox."""
    return sorted({b.bbox.page for b in ir.blocks if b.bbox is not None})


# ============================================================ render seam (impure)


def render_page_gray(pdf_path: Path, page_index: int, dpi: float = _PADDLE_RENDER_DPI):
    """Rasterize 0-based `page_index` of a PDF to a grayscale numpy array at `dpi`.
    Lazy pypdfium2/PIL (already runtime deps). The default 300 dpi matches paddle's
    box space, so IR boxes need no scaling."""
    import numpy as np
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_index]
        try:
            pil = page.render(scale=dpi / 72).to_pil().convert("L")
        finally:
            page.close()
        return np.asarray(pil)
    finally:
        pdf.close()


def document_coverage(
    ir: DocumentIR,
    render_fn: Callable[[int], "object"],
    *,
    dpi: float = _PADDLE_RENDER_DPI,
    box_dpi: float = _PADDLE_RENDER_DPI,
    dilate_px: float = 4.0,
    threshold: int | None = None,
    border_ignore_frac: float = 0.0,
) -> DocCoverage:
    """Coverage for every page carrying a bbox. `render_fn(page_index_0based)`
    returns that page's grayscale array (the injectable render seam — a real
    pypdfium2 render, or a fake in tests). Boxes are scaled from `box_dpi` to `dpi`
    so the mask and the boxes share a pixel space."""
    pages = ir_page_numbers(ir)
    if not pages:
        raise ValueError("IR has no bbox-bearing pages; coverage is undefined")
    out: list[PageCoverage] = []
    for pg in pages:
        gray = render_fn(pg - 1)
        boxes = ir_page_boxes(ir, pg, box_dpi=box_dpi, render_dpi=dpi)
        out.append(page_coverage(
            gray, boxes, page=pg, threshold=threshold,
            dilate_px=dilate_px, border_ignore_frac=border_ignore_frac,
        ))
    return rollup(out)


# ============================================================ additive integration


def annotate_report(report: QualityReport, coverage: DocCoverage) -> QualityReport:
    """Return a copy of `report` with the two coverage fields populated. Additive
    only: `quality_score` and every other field are untouched — coverage is a
    *new* signal alongside the formula, never folded into it by default."""
    return report.model_copy(update={
        "ink_coverage": coverage.ink_coverage,
        "uncovered_ink_ratio": coverage.uncovered_ink_ratio,
    })
