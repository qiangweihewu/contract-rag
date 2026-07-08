import pytest

from contract_rag.demo.highlight import block_rect, fact_highlights
from contract_rag.extract.schema import ContractFacts, ExtractedClause
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR

LETTER = (612.0, 792.0)  # US Letter in PDF points


def _b(engine, bbox, bid="#/b/0", text="Governed by the laws of New York."):
    return DocBlock(block_id=bid, type=BlockType.PARAGRAPH, text=text,
                    bbox=bbox, confidence=1.0, source_engine=engine)


def _ir(blocks):
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", blocks=blocks, metadata={})


def test_paddle_bbox_maps_rendered_pixels_at_300dpi_to_page_fractions():
    # rendered Letter page at 300 dpi = 2550 x 3300 px; a box at the exact center quarter
    b = _b("paddleocr", BoundingBox(page=1, x0=637.5, y0=825.0, x1=1912.5, y1=2475.0))
    r = block_rect(b, [LETTER])
    assert r is not None and r.page == 1
    assert r.left == pytest.approx(0.25) and r.top == pytest.approx(0.25)
    assert r.width == pytest.approx(0.5) and r.height == pytest.approx(0.5)


def test_docling_bbox_maps_points_bottomleft_origin_with_y_flip():
    # docling: points, bottom-left origin — y0 is the TOP edge (larger value).
    # A block spanning y 792→594 (top quarter of the page), x 61.2→550.8.
    b = _b("docling", BoundingBox(page=1, x0=61.2, y0=792.0, x1=550.8, y1=594.0))
    r = block_rect(b, [LETTER])
    assert r is not None
    assert r.left == pytest.approx(0.1) and r.width == pytest.approx(0.8)
    assert r.top == pytest.approx(0.0) and r.height == pytest.approx(0.25)


def test_docling_y_order_is_normalized_either_way():
    swapped = _b("docling", BoundingBox(page=1, x0=61.2, y0=594.0, x1=550.8, y1=792.0))
    r = block_rect(swapped, [LETTER])
    assert r is not None and r.top == pytest.approx(0.0) and r.height == pytest.approx(0.25)


def test_coords_beyond_page_edges_are_clamped():
    b = _b("paddleocr", BoundingBox(page=1, x0=-50.0, y0=-10.0, x1=9000.0, y1=100.0))
    r = block_rect(b, [LETTER])
    assert r is not None
    assert r.left == 0.0 and r.top == 0.0 and r.left + r.width <= 1.0


def test_unusable_boxes_fall_back_to_none():
    assert block_rect(_b("docling", None), [LETTER]) is None                     # no bbox
    assert block_rect(_b("vlm", BoundingBox(page=1, x0=0, y0=0, x1=9, y1=9)), [LETTER]) is None  # unknown convention
    assert block_rect(_b("docling", BoundingBox(page=1, x0=0, y0=0, x1=0, y1=0)), [LETTER]) is None  # degenerate no-prov fallback
    assert block_rect(_b("paddleocr", BoundingBox(page=3, x0=1, y0=1, x1=9, y1=9)), [LETTER]) is None  # page out of range


def test_paddle_multipage_uses_that_pages_size():
    a4 = (595.0, 842.0)
    b = _b("paddleocr", BoundingBox(page=2, x0=0.0, y0=0.0, x1=595.0 * 300 / 72, y1=842.0 * 300 / 72))
    r = block_rect(b, [LETTER, a4])
    assert r is not None and r.page == 2
    assert r.width == pytest.approx(1.0) and r.height == pytest.approx(1.0)


def test_fact_highlights_maps_cited_blocks_and_falls_back_per_fact():
    ir = _ir([
        _b("paddleocr", BoundingBox(page=1, x0=255.0, y0=330.0, x1=2295.0, y1=660.0), bid="#/ocr/0"),
        _b("paddleocr", None, bid="#/ocr/1"),
    ])
    facts = ContractFacts(
        governing_law=ExtractedClause(value="New York", source_block_id="#/ocr/0", confidence=0.9),
        counterparty=ExtractedClause(value="Acme Inc.", source_block_id="#/ocr/1", confidence=0.9),
        effective_date=ExtractedClause(value="2020-01-01", source_block_id="#/missing", confidence=0.9),
    )
    hls = {h.field: h.rect for h in fact_highlights(facts, ir, [LETTER], ContractFacts.FIELD_NAMES)}
    assert hls["governing_law"] is not None and hls["governing_law"].page == 1
    assert hls["governing_law"].left == pytest.approx(0.1)
    assert hls["counterparty"] is None        # cited block has no bbox
    assert hls["effective_date"] is None      # cited block id not in the IR
    assert hls["total_value"] is None         # empty value
