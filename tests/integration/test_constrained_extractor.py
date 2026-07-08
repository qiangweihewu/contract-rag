"""Gated integration test for the `constrained` backend against a live
OpenAI-compatible structured-output server (vLLM / SGLang / Ollama >= 0.5).
Set CONSTRAINED_ENDPOINT — or reuse LOCAL_ENDPOINT, which the backend falls
back to — to enable. Mirrors tests/integration/test_local_extractor.py."""
import os

import pytest

from contract_rag.config import Settings
from contract_rag.extract.constrained import ConstrainedExtractor
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR

ENDPOINT = os.environ.get("CONSTRAINED_ENDPOINT") or os.environ.get("LOCAL_ENDPOINT")


@pytest.mark.skipif(
    not ENDPOINT,
    reason="set CONSTRAINED_ENDPOINT (or LOCAL_ENDPOINT) to a running "
    "OpenAI-compatible structured-output server (vLLM/SGLang/Ollama)",
)
def test_constrained_extractor_extracts_counterparty():
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
    ext = ConstrainedExtractor(
        Settings(
            extract_backend="constrained",
            constrained_endpoint=ENDPOINT,
            constrained_model=os.environ.get(
                "CONSTRAINED_MODEL", os.environ.get("LOCAL_MODEL", "Qwen3-14B")
            ),
        )
    )
    facts = ext.extract(ir)
    assert "Acme" in facts.counterparty.value
    # after the hash-prefix repair, attribution must land on the real block
    assert facts.counterparty.source_block_id == "#/b/1"
    assert ext.last_tokens > 0
