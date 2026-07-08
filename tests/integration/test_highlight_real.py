"""End-to-end provenance highlight on a real OCR parse: reportlab PDF → paddle →
rule extract → report with real pypdfium2 render seams. Skips without paddleocr."""
import functools
from pathlib import Path

import pytest

pytest.importorskip("paddleocr")
pytest.importorskip("pypdfium2")
pytest.importorskip("reportlab")
from reportlab.pdfgen import canvas

from contract_rag.demo.render import page_sizes_pt, render_page_png
from contract_rag.demo.report import build_report_data, render_html
from contract_rag.extract.rules import RuleExtractor
from contract_rag.parse.paddle_parser import parse_with_paddle


def test_scanned_report_highlights_governing_law(tmp_path: Path):
    pdf = tmp_path / "c.pdf"
    c = canvas.Canvas(str(pdf))
    c.drawString(72, 720, "This Agreement shall be governed by the laws of the State of New York.")
    c.save()

    ir = parse_with_paddle(pdf)
    data = build_report_data(
        ir, RuleExtractor(),
        page_sizes=page_sizes_pt(pdf),
        render_page=functools.partial(render_page_png, pdf),
        dirtify_fn=lambda i: i,
    )
    gl = next(f for f in data.fields if f.field == "governing_law")
    assert gl.cleaned_value and gl.highlight is not None
    r = gl.highlight
    # drawString(72, 720) on an A4-ish default page: text sits in the top third
    assert r.page == 1 and 0.0 <= r.top < 0.35 and 0.0 < r.left < 0.3
    assert 0 < r.width <= 1 and 0 < r.height <= 1
    html = render_html(data)
    assert 'class="hl"' in html and "data:image/png;base64," in html
