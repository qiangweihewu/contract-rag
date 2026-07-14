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


from contract_rag.parse.vlm_parser import parse_with_vlm


class _FakeResp:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _settings(**kw):
    from contract_rag.config import Settings
    return Settings(vlm_endpoint="http://fake/v1", **kw)


def test_parse_with_vlm_one_request_per_page_and_model_passthrough(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-fake")
    calls: list[dict] = []

    def fake_post(url, json=None, timeout=None):
        calls.append(json)
        return _FakeResp(f"# Page {len(calls)}\n\ntext {len(calls)}")

    ir = parse_with_vlm(
        pdf,
        _settings(vlm_model="dots.ocr", vlm_prompt="Read the page."),
        post_fn=fake_post,
        render_fn=lambda p: ["b64page1", "b64page2"],
    )
    assert len(calls) == 2                       # one request per page
    assert all(c["model"] == "dots.ocr" for c in calls)
    assert calls[0]["messages"][0]["content"][0]["text"] == "Read the page."
    # exactly one image per request
    assert sum(p["type"] == "image_url" for p in calls[0]["messages"][0]["content"]) == 1
    # page-prefixed unique ids, model-stamped engine, reading order preserved
    assert ir.blocks[0].block_id.startswith("#/vlm/p1/")
    assert any(b.block_id.startswith("#/vlm/p2/") for b in ir.blocks)
    assert all(b.source_engine == "dots.ocr" for b in ir.blocks)
    assert "Page 1" in ir.blocks[0].text


def test_parse_with_vlm_writes_raw_pages_when_dir_set(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-fake")
    raw = tmp_path / "raw"

    ir = parse_with_vlm(
        pdf,
        _settings(vlm_model="m", vlm_raw_dir=raw),
        post_fn=lambda url, json=None, timeout=None: _FakeResp("# H\n\np"),
        render_fn=lambda p: ["only-page"],
    )
    saved = raw / "doc" / "page_0001.md"
    assert saved.read_text() == "# H\n\np"
    assert ir.blocks  # parse still succeeded


def test_post_vlm_request_includes_max_tokens_only_when_set():
    from contract_rag.parse.vlm_parser import _post_vlm_request

    seen = []

    def fake_post(url, json=None, timeout=None):
        seen.append(json)

        class R:
            def raise_for_status(self):
                pass

        return R()

    from contract_rag.config import Settings

    _post_vlm_request(Settings(vlm_endpoint="http://x/v1"), content=[], post_fn=fake_post)
    assert "max_tokens" not in seen[0]  # default None -> payload unchanged
    _post_vlm_request(
        Settings(vlm_endpoint="http://x/v1", vlm_max_tokens=8192), content=[], post_fn=fake_post
    )
    assert seen[1]["max_tokens"] == 8192
