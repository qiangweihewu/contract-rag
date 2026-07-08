from contract_rag.agent.tools import CheckClauseTool, CiteTool, ExtractFieldTool
from contract_rag.ir import BlockType, DocBlock, DocumentIR


def _block(bid, text, btype=BlockType.PARAGRAPH):
    return DocBlock(block_id=bid, type=btype, text=text, confidence=1.0, source_engine="docling")


def _ir():
    return DocumentIR(
        doc_id="d", source_uri="file:///x", file_hash="h", mime_type="application/pdf",
        blocks=[
            _block("b1", "This Agreement is governed by the laws of the State of New York."),
            _block("b2", "Either party may terminate on 30 days notice."),
        ],
        metadata={},
    )


class _FakeExtractor:
    """Returns a ContractFacts-like object with one populated field."""
    def extract(self, ir):
        from contract_rag.extract.schema import ContractFacts, ExtractedClause
        return ContractFacts(
            counterparty=ExtractedClause(),
            effective_date=ExtractedClause(),
            governing_law=ExtractedClause(value="New York", source_block_id="b1", confidence=0.9),
        )


def test_extract_field_tool_pulls_named_field():
    out = ExtractFieldTool(_FakeExtractor(), _ir()).run({"field": "governing_law"})
    assert out == {"field": "governing_law", "value": "New York",
                   "source_block_id": "b1", "confidence": 0.9}


def test_check_clause_tool_finds_termination_block():
    out = CheckClauseTool(_ir()).run({"clause_type": "termination"})
    assert out["present"] is True
    assert "b2" in out["evidence_block_ids"]


def test_cite_tool_returns_block_text():
    out = CiteTool(_ir()).run({"block_id": "b1"})
    assert out["block_id"] == "b1"
    assert "New York" in out["text"]


def test_cite_tool_unknown_block_is_empty_text():
    assert CiteTool(_ir()).run({"block_id": "nope"}) == {"block_id": "nope", "text": ""}
