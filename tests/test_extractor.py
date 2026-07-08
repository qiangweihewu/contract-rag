from contract_rag.extract.extractor import FakeExtractor, build_context, get_extractor
from contract_rag.extract.schema import ContractFacts, ExtractedClause
from contract_rag.config import Settings
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _ir() -> DocumentIR:
    return DocumentIR(
        doc_id="d1", source_uri="file:///x.pdf", file_hash="h", mime_type="application/pdf",
        blocks=[
            DocBlock(block_id="#/b/1", type=BlockType.PARAGRAPH,
                     text="Entered into by Acme Inc.",
                     bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                     confidence=1.0, source_engine="docling")
        ],
        metadata={},
    )


def test_build_context_includes_block_ids():
    ctx = build_context(_ir())
    assert "[#/b/1]" in ctx
    assert "Acme Inc." in ctx


def test_fake_extractor_returns_canned_facts():
    canned = ContractFacts(
        counterparty=ExtractedClause(value="Acme Inc.", source_block_id="#/b/1", confidence=0.9),
        effective_date=ExtractedClause(),
        governing_law=ExtractedClause(),
    )
    facts = FakeExtractor(canned).extract(_ir())
    assert facts.counterparty.value == "Acme Inc."


def test_factory_returns_fake_by_default():
    ex = get_extractor(Settings())  # extract_backend="fake"
    assert isinstance(ex, FakeExtractor)
