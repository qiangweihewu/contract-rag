from __future__ import annotations

from contract_rag.extract.verify import verify
from contract_rag.ir import BlockType, DocBlock, DocumentIR
from contract_rag.verticals.base import ExtractedClause
from contract_rag.verticals.contract.schema import ContractFacts


def _ir() -> DocumentIR:
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", metadata={}, blocks=[
        DocBlock(block_id="b1", type=BlockType.PARAGRAPH, text="governed by New York",
                 confidence=1.0, source_engine="docling")])


def test_verify_passes_attributed_confident_field():
    facts = ContractFacts(
        counterparty=ExtractedClause(),
        effective_date=ExtractedClause(),
        governing_law=ExtractedClause(value="New York", source_block_id="b1", confidence=0.9))
    report = verify(facts, _ir())
    assert report.checks["governing_law"].passed is True
    assert report.checks["governing_law"].attributed is True
    assert report.checks["counterparty"].reasons == ["empty"]


def test_verify_accepts_injected_vertical():
    # red-first: the vertical= keyword does not exist before the rewire
    from contract_rag.verticals.registry import get_vertical
    facts = ContractFacts(
        counterparty=ExtractedClause(),
        effective_date=ExtractedClause(),
        governing_law=ExtractedClause(value="New York", source_block_id="b1", confidence=0.9))
    report = verify(facts, _ir(), vertical=get_vertical("contract"))
    assert report.checks["governing_law"].passed is True
