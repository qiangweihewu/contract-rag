from contract_rag.eval.consistency import (
    consistency_score,
    perturb_whitespace,
)
from contract_rag.extract.schema import ContractFacts, ExtractedClause
from contract_rag.ir import BlockType, DocBlock, DocumentIR


def _ir(text):
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf",
                      blocks=[DocBlock(block_id="b1", type=BlockType.PARAGRAPH, text=text,
                                       confidence=1.0, source_engine="docling")],
                      metadata={})


class _StableExtractor:
    """Always returns the same answer regardless of input -> perfectly consistent."""
    def extract(self, ir):
        return ContractFacts(counterparty=ExtractedClause(value="Acme Inc.", source_block_id="b1"),
                             effective_date=ExtractedClause(), governing_law=ExtractedClause())


class _FlakyExtractor:
    """Echoes the (perturbed) block text into counterparty -> unstable under whitespace noise."""
    def extract(self, ir):
        return ContractFacts(
            counterparty=ExtractedClause(value=ir.blocks[0].text, source_block_id="b1"),
            effective_date=ExtractedClause(), governing_law=ExtractedClause())


def test_perturb_whitespace_is_pure_and_changes_text():
    ir = _ir("hello")
    out = perturb_whitespace(ir)
    assert ir.blocks[0].text == "hello"            # original untouched
    assert out.blocks[0].text != "hello"           # perturbed copy differs
    assert "hello" in out.blocks[0].text.replace(" ", "")  # content preserved


def test_stable_extractor_scores_one():
    res = consistency_score(_StableExtractor(), _ir("Entered into by Acme Inc."))
    assert res["overall"] == 1.0
    assert res["per_field"]["counterparty"] == 1.0


def test_flaky_extractor_is_inconsistent_on_echoed_field():
    res = consistency_score(_FlakyExtractor(), _ir("Entered into by Acme Inc."))
    assert res["per_field"]["counterparty"] == 0.0
    assert res["overall"] < 1.0
