from contract_rag.chunk.chunker import chunk_ir
from contract_rag.chunk.models import Chunk
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _b(text, type=BlockType.PARAGRAPH, bid=None, page=1):
    return DocBlock(block_id=bid or text[:8], type=type, text=text,
                    bbox=BoundingBox(page=page, x0=0, y0=0, x1=1, y1=1),
                    confidence=1.0, source_engine="docling")


def _ir(blocks):
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", blocks=blocks, metadata={})


def test_chunk_groups_body_under_heading_keeping_block_ids():
    ir = _ir([
        _b("Governing Law", type=BlockType.HEADING, bid="h1"),
        _b("This Agreement shall be governed by the laws of New York.", bid="p1"),
        _b("Disputes are resolved in New York courts.", bid="p2"),
    ])
    chunks = chunk_ir(ir, max_chars=500)
    assert len(chunks) == 1
    c = chunks[0]
    assert isinstance(c, Chunk)
    assert c.heading == "Governing Law"
    assert c.block_ids == ["p1", "p2"]
    assert "governed by the laws of New York" in c.text
    assert c.doc_id == "d"


def test_chunk_splits_when_over_max_chars_preserving_all_block_ids():
    long = "word " * 90  # ~450 chars
    ir = _ir([_b("H", type=BlockType.HEADING, bid="h"),
              _b(long, bid="p1"), _b(long, bid="p2"), _b(long, bid="p3")])
    chunks = chunk_ir(ir, max_chars=500)
    assert len(chunks) >= 2
    assert [bid for c in chunks for bid in c.block_ids] == ["p1", "p2", "p3"]


def test_chunk_keeps_table_as_its_own_chunk():
    ir = _ir([_b("Fees", type=BlockType.HEADING, bid="h"),
              _b("Intro paragraph about fees.", bid="p1"),
              _b("| a | b |\n| 1 | 2 |", type=BlockType.TABLE, bid="t1")])
    chunks = chunk_ir(ir, max_chars=5000)
    table = [c for c in chunks if "t1" in c.block_ids]
    assert len(table) == 1 and table[0].block_ids == ["t1"]


def test_chunk_ids_are_unique_and_furniture_skipped():
    ir = _ir([_b("Header noise", type=BlockType.HEADER, bid="hdr"),
              _b("Intro", type=BlockType.HEADING, bid="h"),
              _b("Body one.", bid="p1"), _b("   ", bid="empty")])
    chunks = chunk_ir(ir)
    ids = [bid for c in chunks for bid in c.block_ids]
    assert "hdr" not in ids and "empty" not in ids       # furniture + empty dropped
    assert len({c.chunk_id for c in chunks}) == len(chunks)
