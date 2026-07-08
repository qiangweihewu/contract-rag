from __future__ import annotations

from contract_rag.chunk.models import Chunk
from contract_rag.enrich.enricher import enrich_chunks


def _chunk(text: str) -> Chunk:
    return Chunk(chunk_id="c1", doc_id="d", text=text, block_ids=["b1"])


def test_default_vertical_matches_prior_behavior():
    out = enrich_chunks([_chunk("Confidential and proprietary information.")])
    assert out[0].clause_type == "confidentiality"
    assert "restricted" in out[0].permission_tags


def test_injected_vertical_overrides_classification():
    class StubVertical:
        def classify_clause(self, chunk): return "stub_type"
        def permission_tags(self, chunk): return ["stub_tag"]
    out = enrich_chunks([_chunk("anything")], vertical=StubVertical())
    assert out[0].clause_type == "stub_type"
    assert out[0].permission_tags == ["stub_tag"]
