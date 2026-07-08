from __future__ import annotations

from contract_rag.agent.tools import CheckClauseTool
from contract_rag.ir import BlockType, DocBlock, DocumentIR


def _ir() -> DocumentIR:
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", metadata={}, blocks=[
        DocBlock(block_id="b1", type=BlockType.PARAGRAPH,
                 text="This payment of $500 is due upon invoice.",
                 confidence=1.0, source_engine="docling")])


def test_check_clause_default_vertical():
    out = CheckClauseTool(_ir()).run({"clause_type": "payment"})
    assert out["present"] is True and out["evidence_block_ids"] == ["b1"]


def test_check_clause_injected_vertical():
    class StubVertical:
        def classify_clause(self, chunk): return "custom"
    out = CheckClauseTool(_ir(), vertical=StubVertical()).run({"clause_type": "custom"})
    assert out["present"] is True
