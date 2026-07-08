import os
from pathlib import Path

import pytest

pytest.importorskip("pypdfium2")
pytest.importorskip("reportlab")
from reportlab.pdfgen import canvas

from contract_rag.config import Settings
from contract_rag.parse.vlm_parser import parse_with_vlm

VLM_ENDPOINT = os.environ.get("VLM_ENDPOINT")


@pytest.mark.skipif(not VLM_ENDPOINT, reason="set VLM_ENDPOINT to a running SGLang server")
def test_parse_with_vlm_against_live_endpoint(tmp_path: Path):
    pdf = tmp_path / "c.pdf"
    c = canvas.Canvas(str(pdf))
    c.drawString(72, 720, "Master Services Agreement")
    c.drawString(72, 700, "Governing law: New York.")
    c.save()
    ir = parse_with_vlm(pdf, Settings(vlm_endpoint=VLM_ENDPOINT))
    assert len(ir.blocks) > 0
    assert all(b.source_engine == "unlimited-ocr" for b in ir.blocks)
