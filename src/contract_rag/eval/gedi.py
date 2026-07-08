"""Tobacco800 GEDI groundtruth primitives: parse the signature/logo zone
annotations and map their original-image pixel coordinates onto parsed-block
(rendered-page) pixel coordinates.

Hoisted from `eval/realscan.py` (like `scanio`) so other real-scan harnesses
— the occlusion experiment there and the `eval/signature.py` detector — reuse one
implementation. `realscan` re-exports these names for back-compat. Pure/stdlib only
(xml.etree); no heavy deps.
"""
from __future__ import annotations

from pydantic import BaseModel

from contract_rag.ir import DocBlock

_RENDER_DPI = 300  # paddle_parser renders PDF pages at this dpi


class Zone(BaseModel):
    kind: str  # DLSignature | DLLogo
    x0: float
    y0: float
    x1: float
    y1: float


class PageZones(BaseModel):
    width: float
    height: float
    zones: list[Zone]


def parse_gedi(xml_text: str) -> PageZones:
    """Tobacco800 GEDI groundtruth: DL_PAGE width/height + DL_ZONE col/row/width/height
    in original-image pixels. Namespace-agnostic (matches on tag localname)."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    width = height = 0.0
    zones: list[Zone] = []
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "DL_PAGE":
            width = float(el.get("width", 0))
            height = float(el.get("height", 0))
        elif tag == "DL_ZONE":
            col, row = float(el.get("col", 0)), float(el.get("row", 0))
            w, h = float(el.get("width", 0)), float(el.get("height", 0))
            zones.append(
                Zone(kind=el.get("gedi_type", ""), x0=col, y0=row, x1=col + w, y1=row + h)
            )
    return PageZones(width=width, height=height, zones=zones)


def has_signature_zone(pz: PageZones) -> bool:
    """Ground-truth label: the page is physically signed iff it carries a DLSignature zone."""
    return any(z.kind == "DLSignature" for z in pz.zones)


def zone_scale(img_dpi: float, render_dpi: float = _RENDER_DPI) -> float:
    """Original-image pixels → parsed-block (rendered-page) pixels. `image_to_pdf`
    writes page points = px * 72/dpi; the paddle adapter renders at `render_dpi`,
    so rendered px = original px * render_dpi/dpi."""
    return render_dpi / img_dpi


def scale_zones(zones: list[Zone], s: float) -> list[Zone]:
    return [
        z.model_copy(update={"x0": z.x0 * s, "y0": z.y0 * s, "x1": z.x1 * s, "y1": z.y1 * s})
        for z in zones
    ]


def block_overlaps(block: DocBlock, zones: list[Zone]) -> bool:
    b = block.bbox
    if b is None:
        return False
    return any(b.x0 < z.x1 and z.x0 < b.x1 and b.y0 < z.y1 and z.y0 < b.y1 for z in zones)
