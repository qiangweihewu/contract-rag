from pathlib import Path

import pytest

pytest.importorskip("pypdfium2")
pytest.importorskip("reportlab")
from reportlab.pdfgen import canvas

from contract_rag.eval.rasterize import rasterize_pdf
from contract_rag.parse.probe import probe_document


def _text_pdf(path: Path) -> Path:
    c = canvas.Canvas(str(path))
    c.drawString(72, 720, "This text must NOT survive rasterization.")
    c.save()
    return path


def test_rasterize_strips_text_layer(tmp_path: Path):
    src = _text_pdf(tmp_path / "src.pdf")
    assert probe_document(src).text_coverage == 1.0  # sanity: input has text
    out = rasterize_pdf(src, tmp_path / "scanned.pdf")
    assert probe_document(out).text_coverage == 0.0  # output is image-only
