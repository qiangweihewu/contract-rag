from contract_rag.clean.quality import (
    QualityReport,
    compute_quality_score,
    is_garbled,
    table_integrity,
)
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _block(text, type=BlockType.PARAGRAPH, conf=1.0):
    return DocBlock(block_id="b", type=type, text=text,
                    bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                    confidence=conf, source_engine="docling")


def _ir(blocks):
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", blocks=blocks, metadata={})


def test_is_garbled_flags_mojibake_not_clean_english():
    assert is_garbled("caf\xc3\xa9 \xc3\xa2greement") is True          # mojibake markers
    assert is_garbled("This Agreement is binding.") is False
    assert is_garbled("") is False                          # empty handled elsewhere


def test_table_integrity_is_one_without_tables_and_penalizes_flattened():
    assert table_integrity(_ir([_block("plain text")])) == 1.0
    intact = _ir([_block("| a | b |\n| 1 | 2 |", type=BlockType.TABLE)])
    flat = _ir([_block("a b 1 2", type=BlockType.TABLE)])
    assert table_integrity(intact) == 1.0
    assert table_integrity(flat) < 1.0


def test_quality_score_high_for_clean_low_for_garbled():
    clean = compute_quality_score(_ir([_block("This Agreement is binding."),
                                       _block("Governing law is New York.")]))
    assert clean.quality_score > 0.9
    assert clean.needs_review is False

    dirty = compute_quality_score(_ir([_block("caf\xc3\xa9 \xc3\xa2 \xc3\xa9 \xc3\xa8 \xc3\xa7 \xc3\xb1"),
                                       _block("   "),
                                       _block("\xc3\xa2\xe2\x82\xac\xe2\x80\xa2 \xc3\xa2\xe2\x82\xac\xc5\x93 \xc3\xa2\xe2\x82\xac")]))
    assert dirty.quality_score < 0.75
    assert dirty.needs_review is True


def test_empty_doc_is_zero_score_not_crash():
    r = compute_quality_score(_ir([]))
    assert r.quality_score == 0.0
    assert r.needs_review is True
    assert isinstance(r, QualityReport)
