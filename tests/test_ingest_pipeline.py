from pathlib import Path

from contract_rag.config import Settings
from contract_rag.ingest.pipeline import IngestResult, ingest_document
from contract_rag.ir import BlockType, DocBlock, DocumentIR


def _ir():
    return DocumentIR(
        doc_id="d", source_uri="file:///x", file_hash="h", mime_type="application/pdf",
        blocks=[
            DocBlock(block_id="b1", type=BlockType.PARAGRAPH, text="email jane@acme.com",
                     confidence=1.0, source_engine="docling"),
            DocBlock(block_id="b2", type=BlockType.PARAGRAPH, text="See Section 1.2.3.4.",
                     confidence=1.0, source_engine="docling"),
        ],
        metadata={},
    )


def _fake_parse(_path, _settings):
    return _ir()


def test_ingest_redacts_by_default():
    res = ingest_document(Path("x.pdf"), Settings(), parse_fn=_fake_parse)
    assert isinstance(res, IngestResult)
    assert res.ir.blocks[0].text == "email [REDACTED:EMAIL]"
    # section reference is preserved (Task 1 fix) — not treated as an IP
    assert res.ir.blocks[1].text == "See Section 1.2.3.4."
    assert [m.block_id for m in res.redactions] == ["b1"]


def test_ingest_redact_off_passes_ir_through():
    res = ingest_document(Path("x.pdf"), Settings(redact_pii=False), parse_fn=_fake_parse)
    assert res.ir.blocks[0].text == "email jane@acme.com"
    assert res.redactions == []


def test_explicit_redact_flag_overrides_settings():
    res = ingest_document(Path("x.pdf"), Settings(redact_pii=False),
                          parse_fn=_fake_parse, redact=True)
    assert res.ir.blocks[0].text == "email [REDACTED:EMAIL]"
