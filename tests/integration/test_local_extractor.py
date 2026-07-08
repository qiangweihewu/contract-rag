import os

import pytest

from contract_rag.config import Settings
from contract_rag.extract.extractor import LocalExtractor
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR

LOCAL_ENDPOINT = os.environ.get("LOCAL_ENDPOINT")


@pytest.mark.skipif(
    not LOCAL_ENDPOINT,
    reason="set LOCAL_ENDPOINT to a running OpenAI-compatible server (vLLM/SGLang/Ollama)",
)
def test_local_extractor_extracts_counterparty():
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
    facts = LocalExtractor(
        Settings(
            extract_backend="local",
            local_endpoint=LOCAL_ENDPOINT,
            local_model=os.environ.get("LOCAL_MODEL", "Qwen3-14B"),
        )
    ).extract(ir)
    assert "Acme" in facts.counterparty.value
    assert facts.counterparty.source_block_id == "#/b/1"
