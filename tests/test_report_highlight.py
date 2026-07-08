"""Report ↔ highlight integration: overlay markup appears iff a cited block has a
usable bbox, page images are rendered once and embedded (self-contained), and the
no-bbox path is byte-identical to the pre-highlight report."""
import base64

from contract_rag.demo.report import build_report_data, render_html
from contract_rag.extract.rules import RuleExtractor
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR

LETTER = (612.0, 792.0)

# a 1x1 transparent PNG — the fake render seam's output (no pypdfium2 in unit tests)
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg=="
)


def _b(text, bid, bbox, engine="paddleocr"):
    return DocBlock(block_id=bid, type=BlockType.PARAGRAPH, text=text,
                    bbox=bbox, confidence=1.0, source_engine=engine)


def _ir(blocks):
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", blocks=blocks, metadata={})


def _scan_ir():
    return _ir([
        _b("This Agreement is entered into by and between Acme Inc. and Globex LLC.",
           "#/ocr/0", BoundingBox(page=1, x0=200, y0=300, x1=2300, y1=420)),
        _b("This Agreement shall be governed by the laws of the State of New York.",
           "#/ocr/1", BoundingBox(page=1, x0=200, y0=500, x1=2300, y1=620)),
    ])


def test_report_with_bbox_embeds_page_image_and_overlay():
    calls: list[int] = []

    def render_page(idx: int) -> bytes:
        calls.append(idx)
        return TINY_PNG

    data = build_report_data(_scan_ir(), RuleExtractor(),
                             page_sizes=[LETTER], render_page=render_page,
                             dirtify_fn=lambda i: i)
    assert data.page_images.keys() == {1}
    assert calls == [0]  # both facts cite page 1 → rendered exactly once, reused
    html = render_html(data)
    assert "Source provenance" in html
    assert 'class="hl"' in html
    assert f"data:image/png;base64,{base64.b64encode(TINY_PNG).decode()}" in html
    assert html.count("data:image/png") == 1   # one shared image, self-contained
    assert "· p.1" in html                     # table source cell links to the page


def test_report_without_bbox_renders_exactly_as_before():
    no_box = _ir([
        _b("Governed by the laws of the State of New York.", "#/md/0", None, engine="vlm"),
    ])
    with_seams = build_report_data(no_box, RuleExtractor(),
                                   page_sizes=[LETTER], render_page=lambda i: TINY_PNG)
    without = build_report_data(no_box, RuleExtractor())
    assert with_seams.page_images == {}
    assert render_html(with_seams) == render_html(without)  # zero regression
    assert "Source provenance" not in render_html(without)
    assert 'class="hl"' not in render_html(without)


def test_scanned_flow_skips_simulated_dirt_and_says_so():
    data = build_report_data(_scan_ir(), RuleExtractor(), dirtify_fn=lambda i: i)
    assert data.dirt_simulated is False
    html = render_html(data)
    assert "scanned document as ingested" in html
    assert "simulated enterprise ingestion" not in html


def test_default_flow_still_reports_simulated_dirt():
    html = render_html(build_report_data(_scan_ir(), RuleExtractor()))
    assert "simulated enterprise ingestion" in html
