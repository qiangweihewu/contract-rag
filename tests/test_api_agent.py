from __future__ import annotations

import io

import pytest

pytest.importorskip("fastapi", reason="needs the 'api' extra (fastapi)")

from fastapi.testclient import TestClient

from contract_rag.agent.models import (
    AgentAnswer, AgentResult, AgentState, AgentStatus, Citation,
)
from contract_rag.api.app import create_app
from contract_rag.ir import BlockType, DocBlock, DocumentIR


def _fake_parse(path, settings) -> DocumentIR:
    return DocumentIR(
        doc_id="d", source_uri="file:///x", file_hash="h", mime_type="application/pdf",
        blocks=[DocBlock(block_id="b1", type=BlockType.PARAGRAPH,
                         text="This Agreement is governed by the laws of New York.",
                         confidence=1.0, source_engine="docling")],
        metadata={},
    )


def _fake_agent(ir, task) -> AgentResult:
    state = AgentState(
        task=task,
        status=AgentStatus.DONE,
        answer=AgentAnswer(
            value="New York", confidence=0.8,
            citations=[Citation(block_id="b1", text="governed by the laws of New York")],
        ),
    )
    return AgentResult(state=state, trace_id="trace-xyz")


def _client():
    app = create_app(parse_fn=_fake_parse, agent=_fake_agent)
    return TestClient(app)


def test_agent_endpoint_returns_grounded_answer():
    resp = _client().post(
        "/v1/agent",
        files={"file": ("c.txt", io.BytesIO(b"x"), "text/plain")},
        data={"q": "What law governs?", "field": "governing_law"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "New York"
    assert body["status"] == "done"
    assert body["trace_id"] == "trace-xyz"
    assert body["citations"] == [{"block_id": "b1", "text": "governed by the laws of New York"}]


def test_agent_endpoint_default_is_credential_free():
    # No agent= injected: the real RulePlanner + rule extractor + hashing embedder path
    # must run end-to-end with no secrets/network and return a 200 with a status.
    app = create_app(parse_fn=_fake_parse)
    resp = TestClient(app).post(
        "/v1/agent",
        files={"file": ("c.txt", io.BytesIO(b"x"), "text/plain")},
        data={"q": "What law governs?"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] in {"done", "needs_hitl", "failed"}
