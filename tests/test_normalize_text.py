from contract_rag.clean.normalize import dehyphenate, fix_unicode, strip_whitespace_noise
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _block(text, bid="b", type=BlockType.PARAGRAPH):
    return DocBlock(block_id=bid, type=type, text=text,
                    bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                    confidence=1.0, source_engine="docling")


def _ir(blocks):
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", blocks=blocks, metadata={})


def test_fix_unicode_repairs_mojibake_and_preserves_provenance():
    original = "café—agreement".encode("utf-8").decode("latin-1")  # mojibake
    out = fix_unicode(_ir([_block(original, bid="keep")]))
    assert "café" in out.blocks[0].text
    assert out.blocks[0].block_id == "keep"            # provenance preserved
    assert out.blocks[0].source_engine == "docling"


def test_dehyphenate_joins_line_broken_words():
    out = dehyphenate(_ir([_block("This agree-\nment is binding.")]))
    assert "agreement" in out.blocks[0].text
    assert "agree-" not in out.blocks[0].text


def test_strip_whitespace_collapses_and_drops_empty():
    out = strip_whitespace_noise(_ir([
        _block("  too    much\n\n  space  ", bid="a"),
        _block("   ", bid="empty"),
    ]))
    assert len(out.blocks) == 1                          # empty block dropped
    assert out.blocks[0].text == "too much space"
    assert out.blocks[0].block_id == "a"


def test_strip_whitespace_preserves_table_rows():
    """TABLE blocks must pass through verbatim; newlines separating rows must not be collapsed."""
    table_text = "| a | b |\n| 1 | 2 |"
    out = strip_whitespace_noise(_ir([
        _block(table_text, bid="tbl", type=BlockType.TABLE),
        _block("  collapse   me  ", bid="para"),
    ]))
    # Both blocks survive (table non-empty, paragraph collapses to non-empty)
    assert len(out.blocks) == 2
    # Table text is EXACTLY preserved — newline and row structure intact
    tbl = next(b for b in out.blocks if b.block_id == "tbl")
    assert tbl.text == table_text, f"Table text was mutated: {tbl.text!r}"
    # Non-table paragraph is still collapsed normally
    para = next(b for b in out.blocks if b.block_id == "para")
    assert para.text == "collapse me"
