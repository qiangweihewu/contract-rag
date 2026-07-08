import pytest

from contract_rag.eval.golden import GoldenDoc
from contract_rag.eval.prelabel import (
    PrelabelStatus,
    apply_corrections,
    approve,
    prelabel,
    to_golden,
)
from contract_rag.extract.schema import ContractFacts, ExtractedClause
from contract_rag.ir import BlockType, DocBlock, DocumentIR


def _ir():
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf",
                      blocks=[DocBlock(block_id="b1", type=BlockType.PARAGRAPH,
                                       text="Governed by the laws of the State of New York.",
                                       confidence=1.0, source_engine="docling")],
                      metadata={})


class _Extractor:
    def extract(self, ir):
        return ContractFacts(
            counterparty=ExtractedClause(value="Acme Inc.", source_block_id="b1"),
            effective_date=ExtractedClause(),
            governing_law=ExtractedClause(value="State of New York", source_block_id="b1"))


def test_prelabel_drafts_from_extractor():
    rec = prelabel("doc1", "doc1.pdf", _ir(), _Extractor())
    assert rec.status == PrelabelStatus.DRAFT
    assert rec.draft_facts["counterparty"] == "Acme Inc."
    assert rec.draft_facts["governing_law"] == "State of New York"
    assert set(rec.draft_facts) == set(ContractFacts.FIELD_NAMES)


def test_apply_corrections_is_pure_and_overlays():
    rec = prelabel("doc1", "doc1.pdf", _ir(), _Extractor())
    fixed = apply_corrections(rec, {"effective_date": "January 1, 2020"})
    assert rec.status == PrelabelStatus.DRAFT             # original untouched
    assert fixed.status == PrelabelStatus.CORRECTED
    assert fixed.corrections == {"effective_date": "January 1, 2020"}


def test_to_golden_applies_corrections_and_normalizes():
    rec = prelabel("doc1", "doc1.pdf", _ir(), _Extractor())
    rec = apply_corrections(rec, {"effective_date": "January 1, 2020"})
    gold = to_golden(approve(rec))
    assert isinstance(gold, GoldenDoc)
    assert gold.doc_id == "doc1"
    # governing_law canonicalized to jurisdiction (normalize_facts), corrections win
    assert gold.facts["governing_law"] == "New York"
    assert gold.facts["effective_date"] == "January 1, 2020"


def test_apply_corrections_rejects_unknown_field():
    rec = prelabel("doc1", "doc1.pdf", _ir(), _Extractor())
    with pytest.raises(ValueError, match="governing_laww"):
        apply_corrections(rec, {"governing_laww": "New York"})  # typo'd field name


def test_to_golden_raises_when_a_correction_normalizes_to_empty():
    # "NY" is a real human intent but the jurisdiction canonicalizer can't represent it ->
    # silently empties the field. The gold-labeling workflow must fail loud, not corrupt gold.
    rec = prelabel("doc1", "doc1.pdf", _ir(), _Extractor())
    rec = apply_corrections(rec, {"governing_law": "NY"})
    with pytest.raises(ValueError, match="governing_law"):
        to_golden(approve(rec))
