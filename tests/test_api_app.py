"""Endpoint tests for the FastAPI service, gated on the `api` extra (fastapi).
Uses injected fakes (hand-built IR + FakeExtractor) so no docling/network runs."""
import importlib.util
import io

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None, reason="needs the 'api' extra (fastapi)"
)

from contract_rag.extract.extractor import FakeExtractor
from contract_rag.extract.schema import ContractFacts, ExtractedClause
from contract_rag.ir import BlockType, DocBlock, DocumentIR


def _ir():
    return DocumentIR(
        doc_id="upload", source_uri="file:///x", file_hash="h", mime_type="application/pdf",
        blocks=[
            DocBlock(block_id="b1", type=BlockType.HEADING, text="Governing Law",
                     confidence=1.0, source_engine="docling"),
            DocBlock(block_id="b2", type=BlockType.PARAGRAPH,
                     text="This Agreement is governed by the laws of New York. "
                          "Contact jane@acme.com.",
                     confidence=1.0, source_engine="docling"),
        ],
        metadata={},
    )


def _facts():
    return ContractFacts(
        counterparty=ExtractedClause(),
        effective_date=ExtractedClause(),
        governing_law=ExtractedClause(value="New York", source_block_id="b2", confidence=0.9),
    )


def _client():
    from fastapi.testclient import TestClient

    from contract_rag.api.app import create_app

    app = create_app(parse_fn=lambda _p, _s: _ir(), extractor=FakeExtractor(_facts()))
    return TestClient(app)


def _upload():
    return {"file": ("c.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")}


def test_health_is_ok():
    r = _client().get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_is_ok_for_credentialfree_default():
    r = _client().get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_diagnose_returns_quality_and_redaction_count():
    r = _client().post("/v1/diagnose", files=_upload())
    assert r.status_code == 200
    body = r.json()
    assert "raw_quality" in body and "cleaned_quality" in body
    assert body["redactions"] == 1  # the email is redacted at ingest


def test_extract_returns_verified_facts():
    r = _client().post("/v1/extract", files=_upload())
    assert r.status_code == 200
    body = r.json()
    assert body["facts"]["governing_law"]["value"] == "New York"
    assert body["verification"]["governing_law"]["passed"] is True


def test_ask_returns_sourced_clauses():
    r = _client().post("/v1/ask", files=_upload(), data={"q": "governing law"})
    assert r.status_code == 200
    hits = r.json()["results"]
    assert hits and all("block_ids" in h for h in hits)


def test_metrics_reports_traffic_and_slo():
    client = _client()
    client.post("/v1/diagnose", files=_upload())
    r = client.get("/v1/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["metrics"]["n_traces"] >= 1
    assert "slo" in body and "met" in body["slo"]


def test_root_serves_upload_page():
    r = _client().get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "diagnose" in r.text.lower()


def test_upload_over_size_limit_returns_413():
    from fastapi.testclient import TestClient

    from contract_rag.api.app import create_app
    from contract_rag.config import Settings

    s = Settings(max_upload_mb=1)  # 1MB cap
    app = create_app(settings=s, parse_fn=lambda _p, _s: _ir(), extractor=FakeExtractor(_facts()))
    client = TestClient(app)
    big = io.BytesIO(b"x" * (2 * 1024 * 1024))  # 2MB > 1MB cap
    r = client.post("/v1/diagnose", files={"file": ("big.pdf", big, "application/pdf")})
    assert r.status_code == 413


def test_upload_within_size_limit_still_succeeds():
    from fastapi.testclient import TestClient

    from contract_rag.api.app import create_app
    from contract_rag.config import Settings

    s = Settings(max_upload_mb=1)
    app = create_app(settings=s, parse_fn=lambda _p, _s: _ir(), extractor=FakeExtractor(_facts()))
    client = TestClient(app)
    r = client.post("/v1/diagnose", files=_upload())
    assert r.status_code == 200


def test_parse_failure_returns_structured_400():
    from fastapi.testclient import TestClient

    from contract_rag.api.app import create_app

    def _boom(_p, _s):
        raise ValueError("corrupt PDF")

    app = create_app(parse_fn=_boom, extractor=FakeExtractor(_facts()))
    client = TestClient(app)
    r = client.post("/v1/diagnose", files=_upload())
    assert r.status_code == 400
    assert "corrupt PDF" in r.json()["detail"]


def test_app_boots_for_gated_backend_and_ready_reports_503():
    # A misconfigured deploy (openai backend, gate off) must not crash startup:
    # the extractor is built lazily, so the app boots and /ready reports 503.
    from fastapi.testclient import TestClient

    from contract_rag.api.app import create_app
    from contract_rag.config import Settings

    s = Settings(extract_backend="openai", allow_external_llm=False)
    app = create_app(settings=s, parse_fn=lambda _p, _s: _ir())  # must not raise
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["ready"] is False
