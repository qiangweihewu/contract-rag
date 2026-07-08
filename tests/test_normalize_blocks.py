from contract_rag.clean.normalize import dedupe_blocks, strip_headers_footers
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _b(text, type=BlockType.PARAGRAPH, page=1, bid=None):
    return DocBlock(block_id=bid or text[:6], type=type, text=text,
                    bbox=BoundingBox(page=page, x0=0, y0=0, x1=1, y1=1),
                    confidence=1.0, source_engine="docling")


def _ir(blocks):
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", blocks=blocks, metadata={})


def test_strip_headers_footers_drops_typed_blocks():
    out = strip_headers_footers(_ir([
        _b("Page 1 of 9", type=BlockType.HEADER, bid="h"),
        _b("Real clause text.", bid="body"),
        _b("Confidential", type=BlockType.FOOTER, bid="f"),
    ]))
    ids = [b.block_id for b in out.blocks]
    assert ids == ["body"]


def test_strip_headers_footers_drops_repeated_page_furniture():
    blocks = []
    for pg in range(1, 5):  # 4 pages, "ACME INC" on every page = furniture
        blocks.append(_b("ACME INC", page=pg, bid=f"hdr{pg}"))
        blocks.append(_b(f"Unique clause on page {pg}.", page=pg, bid=f"body{pg}"))
    out = strip_headers_footers(_ir(blocks))
    texts = [b.text for b in out.blocks]
    assert "ACME INC" not in texts                       # repeated furniture dropped
    assert any("Unique clause on page 3" in t for t in texts)  # real content kept


def test_strip_headers_footers_keeps_long_repeated_clause():
    long_clause = "This long recurring contractual clause appears on multiple pages verbatim and must be kept."
    blocks = [_b(long_clause, page=p, bid=f"c{p}") for p in range(1, 4)]
    out = strip_headers_footers(_ir(blocks))
    assert len(out.blocks) == 3                          # long text (>80) is not furniture


def test_dedupe_removes_near_duplicate_blocks():
    out = dedupe_blocks(_ir([
        _b("The parties agree to the following terms and conditions.", bid="orig"),
        _b("The parties agree to the following terms and conditions.", bid="dup"),
        _b("A completely different sentence about governing law.", bid="other"),
    ]))
    ids = [b.block_id for b in out.blocks]
    assert ids == ["orig", "other"]                      # first kept, dup dropped
