"""Shared scan-input helpers: wrap page images in single-page PDFs so the parse
router can ingest them. Hoisted from `eval/realscan.py` so other real-scan harnesses
(e.g. `eval/fincritical.py`) reuse one implementation. PIL is lazy-imported so the
unit suite runs without it."""
from __future__ import annotations

from pathlib import Path

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def image_dpi(img_path: Path, default: float = 150.0) -> float:
    from PIL import Image

    with Image.open(img_path) as im:
        dpi = im.info.get("dpi")
    if not dpi:
        return default
    val = float(dpi[0] if isinstance(dpi, (tuple, list)) else dpi)
    return val if val > 1 else default  # PIL reports 1.0 for "unset"


def image_to_pdf(img_path: Path, out_pdf: Path, default_dpi: float = 150.0) -> Path:
    """Wrap a scan image in a single-page PDF at its native resolution, so page points
    = pixels * 72/dpi and a later render at the parser's render dpi maps original
    pixel coords by exactly `realscan.zone_scale(dpi)`."""
    from PIL import Image

    dpi = image_dpi(img_path, default=default_dpi)
    with Image.open(img_path) as im:
        im.convert("RGB").save(out_pdf, "PDF", resolution=dpi)
    return Path(out_pdf)


def ensure_pdf(doc: Path, pdf_cache_dir: Path) -> Path:
    """PDF passthrough; images converted once and cached by stem."""
    doc = Path(doc)
    if doc.suffix.lower() == ".pdf":
        return doc
    pdf_cache_dir = Path(pdf_cache_dir)
    pdf_cache_dir.mkdir(parents=True, exist_ok=True)
    out = pdf_cache_dir / f"{doc.stem}.pdf"
    if not out.exists():
        image_to_pdf(doc, out)
    return out
