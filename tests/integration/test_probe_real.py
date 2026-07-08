from pathlib import Path

import pytest

pytest.importorskip("pypdfium2")
pytest.importorskip("reportlab")
from PIL import Image
from reportlab.pdfgen import canvas

from contract_rag.parse.probe import probe_document


def _text_pdf(path: Path) -> Path:
    c = canvas.Canvas(str(path))
    c.drawString(72, 720, "This is a digital contract with a text layer.")
    c.save()
    return path


def _image_only_pdf(path: Path) -> Path:
    Image.new("RGB", (612, 792), "white").save(str(path), "PDF")
    return path


def test_probe_distinguishes_text_from_imageonly(tmp_path: Path):
    assert probe_document(_text_pdf(tmp_path / "t.pdf")).text_coverage == 1.0
    assert probe_document(_image_only_pdf(tmp_path / "i.pdf")).text_coverage == 0.0
