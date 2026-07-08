from __future__ import annotations

from contract_rag.chunk.models import Chunk
from contract_rag.verticals.nda.enrich import classify_clause, permission_tags


def _c(text: str) -> Chunk:
    return Chunk(chunk_id="c1", doc_id="d", text=text, block_ids=["b1"])


def test_classify_and_tags():
    assert classify_clause(_c("All Confidential Information shall be kept confidential.")) == "confidentiality"
    assert classify_clause(_c("Receiving Party shall return or destroy all materials.")) == "return_of_materials"
    assert classify_clause(_c("This Agreement is governed by the laws of New York.")) == "governing_law"
    assert "restricted" in permission_tags(_c("proprietary and confidential information"))
