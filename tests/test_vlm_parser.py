import pytest

from contract_rag.config import Settings
from contract_rag.parse.vlm_parser import (
    _extract_markdown,
    _post_vlm_request,
    build_vlm_ir,
    parse_with_vlm,
)

MD = "# Title\n\nA paragraph from the VLM.\n"


class _FakeResponse:
    def __init__(self, payload=None, status_error=None):
        self._payload = payload
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error:
            raise self._status_error

    def json(self):
        return self._payload


def _ok_response(content="hello"):
    return _FakeResponse({"choices": [{"message": {"content": content}}]})


def test_build_vlm_ir_wraps_markdown_blocks():
    ir = build_vlm_ir(doc_id="d1", source_uri="file:///x.pdf", file_hash_str="h", markdown=MD)
    assert ir.doc_id == "d1"
    assert ir.mime_type == "application/pdf"
    assert len(ir.blocks) >= 2
    assert all(b.source_engine == "unlimited-ocr" for b in ir.blocks)


def test_parse_with_vlm_requires_endpoint(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="vlm_endpoint"):
        parse_with_vlm(pdf, Settings(vlm_endpoint=None))


# ---------------------------------------------------------- retry / timeout / parse


def test_post_vlm_request_uses_configured_timeout():
    calls = []

    def fake_post(url, json, timeout):
        calls.append(timeout)
        return _ok_response()

    settings = Settings(vlm_endpoint="http://vlm:8000/v1", vlm_timeout=42)
    _post_vlm_request(settings, content=[], post_fn=fake_post)
    assert calls == [42]


def test_post_vlm_request_retries_once_then_succeeds():
    attempts = []

    def flaky_post(url, json, timeout):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("network blip")
        return _ok_response("recovered")

    settings = Settings(vlm_endpoint="http://vlm:8000/v1")
    resp = _post_vlm_request(settings, content=[], post_fn=flaky_post, sleep_fn=lambda s: None)
    assert len(attempts) == 2
    assert _extract_markdown(resp) == "recovered"


def test_post_vlm_request_raises_after_exhausting_retries():
    def always_fails(url, json, timeout):
        raise ConnectionError("down")

    settings = Settings(vlm_endpoint="http://vlm:8000/v1")
    with pytest.raises(ConnectionError, match="failed after"):
        _post_vlm_request(settings, content=[], post_fn=always_fails, sleep_fn=lambda s: None)


def test_extract_markdown_returns_content():
    assert _extract_markdown(_ok_response("hi there")) == "hi there"


def test_extract_markdown_raises_clear_error_on_malformed_shape():
    resp = _FakeResponse({"unexpected": "shape"})
    with pytest.raises(ValueError, match="unexpected VLM response shape"):
        _extract_markdown(resp)
