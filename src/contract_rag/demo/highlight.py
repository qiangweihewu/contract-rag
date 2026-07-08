"""Bbox → page-overlay mapping for source-provenance highlighting.

Each parse engine stamps `DocBlock.bbox` in its own coordinate system (verified
against real parsed IRs, not assumed from docs):

- ``paddleocr`` — rendered-page **pixels** at 300 dpi (`paddle_parser` renders
  PDF pages at that dpi), top-left origin (y grows downward), `page` 1-based.
- ``docling`` — PDF **points**, bottom-left origin (y grows upward, so y0 is the
  *top* edge and numerically larger than y1); items without provenance carry a
  degenerate ``(0, 0, 0, 0)`` box.
- the VLM path loses geometry in the markdown round-trip (`markdown_ir` stamps
  ``bbox=None``) — those facts fall back to text-only source display.

Everything here normalizes to **page fractions** in [0, 1] (left/top/width/height,
top-left origin), so an overlay div positioned with CSS percentages is correct at
whatever dpi the page image is later rendered. Granularity is the parser's block
(paddle: an OCR line; docling: a layout item) — not sub-span. Pure: no I/O; page
sizes (PDF points) are injected (see `demo.render` for the pypdfium2 seam).
"""
from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel

from contract_rag.ir import DocBlock, DocumentIR

_PADDLE_RENDER_DPI = 300.0  # paddle_parser renders PDF pages at this dpi


class HighlightRect(BaseModel):
    """A block's position on its page, as fractions of page width/height (top-left)."""

    page: int  # 1-based, matching BoundingBox.page
    left: float
    top: float
    width: float
    height: float


class FactHighlight(BaseModel):
    field: str
    rect: HighlightRect | None  # None = cited block has no usable bbox (fallback)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def block_rect(block: DocBlock, page_sizes_pt: Sequence[tuple[float, float]]) -> HighlightRect | None:
    """Map a block's bbox to page-fraction overlay coords, or None when there is no
    usable box: missing bbox, an engine whose convention we don't know, a page index
    outside the document, or a degenerate/zero-area box (docling's no-provenance
    fallback). Never guesses — an un-mappable block is a fallback, not a wrong box."""
    b = block.bbox
    if b is None or b.page < 1 or b.page > len(page_sizes_pt):
        return None
    w_pt, h_pt = page_sizes_pt[b.page - 1]
    if w_pt <= 0 or h_pt <= 0:
        return None
    if block.source_engine == "paddleocr":
        # rendered px at 300 dpi, top-left origin; rendered page = points * dpi/72 px
        pw = w_pt * _PADDLE_RENDER_DPI / 72
        ph = h_pt * _PADDLE_RENDER_DPI / 72
        left, right = b.x0 / pw, b.x1 / pw
        top, bottom = b.y0 / ph, b.y1 / ph
    elif block.source_engine == "docling":
        # points, bottom-left origin: larger y = closer to the page top → flip
        left, right = b.x0 / w_pt, b.x1 / w_pt
        top = 1 - max(b.y0, b.y1) / h_pt
        bottom = 1 - min(b.y0, b.y1) / h_pt
    else:
        return None  # vlm/markdown and synthetic engines carry no page geometry
    left, top, right, bottom = _clamp(left), _clamp(top), _clamp(right), _clamp(bottom)
    if right <= left or bottom <= top:
        return None
    return HighlightRect(page=b.page, left=left, top=top, width=right - left, height=bottom - top)


def fact_highlights(
    facts, ir: DocumentIR, page_sizes_pt: Sequence[tuple[float, float]],
    field_names: Sequence[str],
) -> list[FactHighlight]:
    """One descriptor per fact field (vertical-agnostic: `field_names` from the
    vertical, `facts` any model whose fields are ExtractedClause-shaped). Empty
    values, unknown block ids, and bbox-less blocks all yield rect=None."""
    blocks = {b.block_id: b for b in ir.blocks}
    out: list[FactHighlight] = []
    for name in field_names:
        clause = getattr(facts, name)
        rect = None
        if clause.value and clause.source_block_id in blocks:
            rect = block_rect(blocks[clause.source_block_id], page_sizes_pt)
        out.append(FactHighlight(field=name, rect=rect))
    return out
