from __future__ import annotations

from contract_rag.chunk.models import Chunk


def _chunk(text: str) -> Chunk:
    return Chunk(chunk_id="c1", doc_id="d", text=text, block_ids=["b1"])


def test_classify_clause_importable_from_both_paths():
    from contract_rag.enrich.enricher import classify_clause as shim
    from contract_rag.verticals.contract.enrich import classify_clause as canon
    assert shim is canon


def test_classify_and_tags_unchanged():
    from contract_rag.verticals.contract.enrich import classify_clause, permission_tags
    c = _chunk("This payment of $500 is due on invoice.")
    assert classify_clause(c) == "payment"
    assert "finance" in permission_tags(c)
