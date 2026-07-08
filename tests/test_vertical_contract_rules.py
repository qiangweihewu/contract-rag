from __future__ import annotations

from contract_rag.ir import BlockType, DocBlock, DocumentIR


def _b(bid: str, text: str):
    return DocBlock(block_id=bid, type=BlockType.PARAGRAPH, text=text,
                    confidence=1.0, source_engine="docling")


def _ir() -> DocumentIR:
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", metadata={}, blocks=[
        _b("b1", "This Agreement is made by and between Acme Inc. and Beta Corp."),
        _b("b2", "This Agreement shall be governed by the laws of the State of New York."),
    ])


def test_rules_importable_from_both_paths():
    from contract_rag.extract.rules import RuleExtractor as Shim, _DATE  # noqa: F401
    from contract_rag.verticals.contract.rules import RuleExtractor as Canon
    assert Shim is Canon


def test_rule_extractor_output_unchanged():
    from contract_rag.verticals.contract.rules import RuleExtractor
    facts = RuleExtractor().extract(_ir())
    assert facts.governing_law.value == "New York"
    assert facts.governing_law.source_block_id == "b2"
    assert "Acme Inc" in facts.counterparty.value
    assert facts.counterparty.source_block_id == "b1"
