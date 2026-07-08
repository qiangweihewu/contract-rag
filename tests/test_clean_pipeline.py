from contract_rag.clean.pipeline import DEFAULT_STEPS, clean_ir
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _b(text, type=BlockType.PARAGRAPH, bid=None):
    return DocBlock(block_id=bid or text[:6], type=type, text=text,
                    bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                    confidence=1.0, source_engine="docling")


def _ir(blocks):
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", blocks=blocks, metadata={})


def test_default_steps_order():
    names = [s.__name__ for s in DEFAULT_STEPS]
    assert names == ["fix_unicode", "dehyphenate", "strip_headers_footers",
                     "dedupe_blocks", "strip_whitespace_noise"]


def test_clean_ir_runs_all_steps():
    ir = _ir([
        _b("Page 1", type=BlockType.HEADER, bid="hdr"),
        _b("This agree-\nment is binding.", bid="body"),
        _b("This agree-\nment is binding.", bid="dup"),
        _b("   ", bid="empty"),
    ])
    out = clean_ir(ir)
    ids = [b.block_id for b in out.blocks]
    assert "hdr" not in ids       # header dropped
    assert "empty" not in ids     # empty dropped
    assert "dup" not in ids       # duplicate dropped
    assert ids == ["body"]
    assert "agreement is binding." in out.blocks[0].text  # dehyphenated


def test_clean_ir_accepts_custom_steps():
    from contract_rag.clean.normalize import strip_whitespace_noise
    ir = _ir([_b("  x   y  ", bid="a")])
    out = clean_ir(ir, steps=[strip_whitespace_noise])
    assert out.blocks[0].text == "x y"
