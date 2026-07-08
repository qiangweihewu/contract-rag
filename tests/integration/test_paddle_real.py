from pathlib import Path

import pytest

pytest.importorskip("paddleocr")
pytest.importorskip("pypdfium2")
pytest.importorskip("reportlab")
from reportlab.pdfgen import canvas

from contract_rag.parse.paddle_parser import parse_with_paddle


def test_parse_with_paddle_reads_text(tmp_path: Path):
    pdf = tmp_path / "c.pdf"
    c = canvas.Canvas(str(pdf))
    c.drawString(72, 720, "GOVERNING LAW NEW YORK")
    c.save()
    ir = parse_with_paddle(pdf)
    assert any(b.source_engine == "paddleocr" for b in ir.blocks)
