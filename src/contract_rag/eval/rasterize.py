from __future__ import annotations

from pathlib import Path


def rasterize_pdf(path: Path, out_path: Path, dpi: int = 200) -> Path:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        images = []
        for page in pdf:
            try:
                images.append(page.render(scale=dpi / 72).to_pil().convert("RGB"))
            finally:
                page.close()
    finally:
        pdf.close()
    if not images:
        raise ValueError(f"no pages to rasterize in {path}")
    images[0].save(out_path, "PDF", save_all=True, append_images=images[1:])
    return Path(out_path)
