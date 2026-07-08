from __future__ import annotations

import contract_rag.verticals.registry as _reg

from contract_rag.chunk.models import Chunk
from contract_rag.enrich.enricher import enrich_chunks
from contract_rag.eval.golden import GoldenDoc
from contract_rag.eval.metrics import aggregate, row_for
from contract_rag.ir import BlockType, DocBlock, DocumentIR
from contract_rag.verticals.base import Vertical
from contract_rag.verticals.registry import get_vertical, register_vertical

from tests.verticals.memo import MemoVertical


def test_new_vertical_runs_end_to_end_with_no_core_fork():
    # Snapshot the registry so this test doesn't leave `memo` registered globally.
    _snapshot = dict(_reg._REGISTRY)
    try:
        # 1. Register a brand-new vertical via the public extension point only.
        memo = MemoVertical()
        register_vertical(memo)
        assert isinstance(memo, Vertical)
        assert get_vertical("memo") is memo

        # 2. Enrich uses the memo taxonomy through the generic engine.
        chunks = [Chunk(chunk_id="c1", doc_id="m", text="From: Jane Doe", block_ids=["b1"])]
        enriched = enrich_chunks(chunks, vertical=memo)
        assert enriched[0].clause_type == "header"
        assert enriched[0].permission_tags == ["internal"]

        # 3. The memo rule extractor produces MemoFacts.
        ir = DocumentIR(doc_id="m", source_uri="file:///x", file_hash="h",
                        mime_type="application/pdf", metadata={}, blocks=[
            DocBlock(block_id="b1", type=BlockType.PARAGRAPH, text="From: Jane Doe",
                     confidence=1.0, source_engine="docling"),
            DocBlock(block_id="b2", type=BlockType.PARAGRAPH, text="Dated 2026-01-15 internally.",
                     confidence=1.0, source_engine="docling")])
        pred = memo.rule_extractor.extract(ir)
        assert pred.author.value == "Jane Doe"
        assert pred.date.value == "2026-01-15"

        # 4. The generic metric engine scores it using the memo's field metadata.
        gold = GoldenDoc(doc_id="m", source_pdf="m.pdf",
                         facts={"author": "Jane Doe", "date": "2026-01-15"})
        agg = aggregate([row_for(pred, gold, ir, memo)], vertical=memo)
        assert agg["n_docs"] == 1
        assert agg["per_field"]["author"] == 1.0
        assert agg["per_field"]["date"] == 1.0
        assert agg["source_accuracy"] == 1.0
    finally:
        _reg._REGISTRY.clear()
        _reg._REGISTRY.update(_snapshot)
