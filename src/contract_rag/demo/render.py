"""Page-render helpers for the demo report (lazy pypdfium2 — already a runtime dep
via `parse.probe`). Kept out of `demo.highlight` so the mapping logic stays pure;
`report.build_report_data` takes these as injectable seams, unit tests pass fakes.
"""
from __future__ import annotations

import io
from pathlib import Path


def page_sizes_pt(pdf_path: Path) -> list[tuple[float, float]]:
    """(width, height) in PDF points for every page, in page order."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        sizes: list[tuple[float, float]] = []
        for page in pdf:
            try:
                sizes.append(page.get_size())
            finally:
                page.close()
        return sizes
    finally:
        pdf.close()


def render_page_png(pdf_path: Path, page_index: int, dpi: float = 144.0) -> bytes:
    """Render 0-based page `page_index` to PNG bytes at `dpi` (144 keeps the
    self-contained report readable without ballooning it)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_index]
        try:
            pil = page.render(scale=dpi / 72).to_pil()
        finally:
            page.close()
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()
    finally:
        pdf.close()
