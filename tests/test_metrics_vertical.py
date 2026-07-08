from __future__ import annotations

from contract_rag.eval.golden import GoldenDoc
from contract_rag.eval.metrics import aggregate, field_scores, row_for
from contract_rag.ir import BlockType, DocBlock, DocumentIR
from contract_rag.verticals.base import ExtractedClause
from contract_rag.verticals.contract.schema import ContractFacts


def _pred() -> ContractFacts:
    return ContractFacts(
        counterparty=ExtractedClause(value="Acme Inc.", source_block_id="b1", confidence=0.7),
        effective_date=ExtractedClause(),
        governing_law=ExtractedClause(value="New York", source_block_id="b1", confidence=0.7),
    )


def _gold() -> GoldenDoc:
    return GoldenDoc(doc_id="d", source_pdf="d.pdf",
                     facts={"governing_law": "New York", "counterparty": "Acme Inc."})


def _ir() -> DocumentIR:
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", metadata={}, blocks=[
        DocBlock(block_id="b1", type=BlockType.PARAGRAPH,
                 text="Acme Inc. governed by New York",
                 confidence=1.0, source_engine="docling")])


def test_field_scores_equivalence():
    s = field_scores(_pred(), _gold())
    assert s["governing_law"] is True
    assert s["counterparty"] is True
    assert s["effective_date"] is False


def test_aggregate_uses_injected_vertical_field_names():
    from contract_rag.verticals.registry import get_vertical
    v = get_vertical("contract")
    # passing vertical= (and the 4th row_for arg) is the red-first new capability
    agg = aggregate([row_for(_pred(), _gold(), _ir(), v)], vertical=v)
    assert agg["n_docs"] == 1
    assert agg["source_accuracy"] == 1.0
    assert set(agg["per_field"]) == set(ContractFacts.FIELD_NAMES)
