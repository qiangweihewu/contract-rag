"""End-to-end with zero secrets/network: rule extractor + hashing-embedder index."""
from contract_rag.agent.models import AgentStatus, AgentTask
from contract_rag.agent.planner import RulePlanner
from contract_rag.agent.runner import build_agent_tools, run_agent
from contract_rag.chunk.models import Chunk
from contract_rag.config import Settings
from contract_rag.extract.extractor import get_extractor
from contract_rag.index.hybrid import build_index
from contract_rag.ir import BlockType, DocBlock, DocumentIR
from contract_rag.obs.store import InMemoryTraceStore
from contract_rag.obs.tracer import Tracer


def _ir():
    return DocumentIR(
        doc_id="d", source_uri="file:///x", file_hash="h", mime_type="application/pdf",
        blocks=[
            DocBlock(block_id="b1", type=BlockType.PARAGRAPH, confidence=1.0, source_engine="docling",
                     text="This Agreement shall be governed by the laws of the State of New York."),
            DocBlock(block_id="b2", type=BlockType.PARAGRAPH, confidence=1.0, source_engine="docling",
                     text="Either party may terminate this Agreement upon 30 days written notice."),
        ],
        metadata={},
    )


def _chunks(ir):
    return [
        Chunk(chunk_id=b.block_id, doc_id=ir.doc_id, text=b.text, block_ids=[b.block_id],
              permission_tags=["general"])
        for b in ir.blocks
    ]


def test_agent_end_to_end_credential_free():
    ir = _ir()
    index = build_index(_chunks(ir))                       # hashing embedder (default)
    extractor = get_extractor(Settings(extract_backend="rule"))  # no creds
    tools = build_agent_tools(ir, index, extractor)
    store = InMemoryTraceStore()

    result = run_agent(AgentTask(question="What is the governing law?", doc_id="d"),
                       RulePlanner(), tools, tracer=Tracer(store=store))

    # the rule extractor finds the New York jurisdiction in b1
    assert result.state.answer is not None
    assert "New York" in result.state.answer.value
    assert result.state.status in (AgentStatus.DONE, AgentStatus.NEEDS_HITL)  # depends on rule confidence
    # full reasoning trace captured
    assert [s.name for s in store.all()[0].spans][:1] == ["retrieve"]
    assert "extract_field" in [s.name for s in store.all()[0].spans]
