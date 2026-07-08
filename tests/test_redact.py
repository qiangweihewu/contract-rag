from contract_rag.ir import BlockType, DocBlock, DocumentIR
from contract_rag.security.pii import PIIType
from contract_rag.security.redact import RedactionResult, redact_ir, redact_text


def _block(block_id, text):
    return DocBlock(block_id=block_id, type=BlockType.PARAGRAPH, text=text,
                    confidence=1.0, source_engine="docling")


def _ir(blocks):
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", blocks=blocks, metadata={})


def test_redact_text_masks_email_and_returns_match():
    out, matches = redact_text("write to jane@acme.com now")
    assert out == "write to [REDACTED:EMAIL] now"
    assert [m.type for m in matches] == [PIIType.EMAIL]


def test_redact_ir_is_pure_and_stamps_block_id():
    ir = _ir([_block("b1", "email jane@acme.com"), _block("b2", "no pii here")])
    result = redact_ir(ir)
    assert isinstance(result, RedactionResult)
    # original untouched (immutability)
    assert ir.blocks[0].text == "email jane@acme.com"
    # redacted copy
    assert result.ir.blocks[0].text == "email [REDACTED:EMAIL]"
    assert result.ir.blocks[1].text == "no pii here"
    assert [m.block_id for m in result.matches] == ["b1"]


def test_redact_ir_no_pii_returns_equivalent_ir():
    ir = _ir([_block("b1", "This Agreement is binding.")])
    result = redact_ir(ir)
    assert result.matches == []
    assert result.ir.blocks[0].text == "This Agreement is binding."


def test_redact_text_handles_overlapping_matches_without_leak():
    # IP and phone spans overlap; merged into one region. Exact-match so this test
    # fails on the pre-fix reverse-splice (which mangled the label) — a real guard.
    out, matches = redact_text("host 192.168.100.200 555-0199 end")
    assert out == "host [REDACTED:IP] end"
    assert "192.168.100.200" not in out and "555-0199" not in out


def test_redact_text_disjoint_multi_match_preserved():
    out, matches = redact_text("from alice@x.com to 123-45-6789")
    assert out == "from [REDACTED:EMAIL] to [REDACTED:SSN]"
    assert len(matches) == 2
