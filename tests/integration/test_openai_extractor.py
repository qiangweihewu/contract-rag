import os

import pytest

from contract_rag.config import Settings
from contract_rag.extract.extractor import OpenAIExtractor
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


@pytest.mark.skipif(
    not (os.environ.get("OPENAI_API_KEY") and os.environ.get("ALLOW_EXTERNAL_LLM")),
    reason="needs OPENAI_API_KEY and ALLOW_EXTERNAL_LLM=true",
)
def test_openai_extractor_extracts_counterparty():
    ir = DocumentIR(
        doc_id="d", source_uri="file:///x", file_hash="h", mime_type="application/pdf",
        blocks=[
            DocBlock(block_id="#/b/1", type=BlockType.PARAGRAPH,
                     text="This Agreement is entered into by Acme Inc. and Globex LLC. "
                          "This Agreement shall be governed by the laws of the State of New York.",
                     bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                     confidence=1.0, source_engine="docling")
        ],
        metadata={},
    )
    facts = OpenAIExtractor(Settings(extract_backend="openai", allow_external_llm=True)).extract(ir)
    assert "Acme" in facts.counterparty.value
    assert facts.counterparty.source_block_id == "#/b/1"
