from __future__ import annotations

from pathlib import Path

import pytest

from contract_rag.config import Settings
from contract_rag.ir import BlockType
from contract_rag.parse.franken_parser import (
    _run_focr,
    franken_json_to_blocks,
    franken_layout_by_page,
    parse_with_franken,
)
from contract_rag.parse.markdown_ir import markdown_to_blocks
from contract_rag.parse.probe import DocProfile
from contract_rag.parse.router import route


# ---------------------------------------------------------------- franken_json_to_blocks


def test_single_page_payload_produces_page_1_blocks():
    payload = {
        "schema_version": 1,
        "markdown": "# Title\n\nSome paragraph text.",
        "layout": [{"label": "title", "boxes": [[0, 0, 10, 10]]}],
    }
    blocks = franken_json_to_blocks(payload, n_pages=1)

    assert len(blocks) == 2
    assert all(b.source_engine == "frankenocr" for b in blocks)
    assert all(b.confidence == 1.0 for b in blocks)
    ids = [b.block_id for b in blocks]
    assert ids == ["#/franken/p1/0", "#/franken/p1/1"]
    assert len(set(ids)) == len(ids)


def test_multi_page_payload_splits_on_page_separator_and_ids_stay_unique():
    payload = {
        "schema_version": 1,
        "markdown": (
            "# Page1 Heading\n\nPara one text.\n"
            "<PAGE>\n"
            "# Page2 Heading\n\nPara two text.\n"
            "<PAGE>\n"
            "# Page3 Heading\n\nPara three text."
        ),
        "pages": [
            {"page": 1, "layout": []},
            {"page": 2, "layout": []},
            {"page": 3, "layout": []},
        ],
    }
    blocks = franken_json_to_blocks(payload, n_pages=3)

    ids = [b.block_id for b in blocks]
    assert ids == [
        "#/franken/p1/0",
        "#/franken/p1/1",
        "#/franken/p2/0",
        "#/franken/p2/1",
        "#/franken/p3/0",
        "#/franken/p3/1",
    ]
    assert len(set(ids)) == 6
    assert [b.text for b in blocks if b.type is BlockType.TITLE] == [
        "Page1 Heading",
        "Page2 Heading",
        "Page3 Heading",
    ]
    assert all(b.source_engine == "frankenocr" for b in blocks)


def test_separator_count_mismatch_falls_back_to_single_page():
    # n_pages says 3 but the markdown only carries one separator (2 segments) —
    # must not raise, and must land everything on page 1.
    payload = {"schema_version": 1, "markdown": "Para A text.\n<PAGE>\nPara B text."}
    blocks = franken_json_to_blocks(payload, n_pages=3)

    assert len(blocks) >= 1
    assert all(b.block_id.startswith("#/franken/p1/") for b in blocks)


def test_headings_lists_and_tables_produce_expected_block_types():
    md = (
        "# Title\n\n"
        "## Heading Two\n\n"
        "A paragraph.\n\n"
        "- list item one\n\n"
        "| a | b |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n"
    )
    expected_types = [b.type for b in markdown_to_blocks(md)]
    blocks = franken_json_to_blocks({"markdown": md}, n_pages=1)
    assert [b.type for b in blocks] == expected_types
    assert BlockType.TITLE in expected_types
    assert BlockType.HEADING in expected_types
    assert BlockType.LIST_ITEM in expected_types
    assert BlockType.TABLE in expected_types


# ---------------------------------------------------------------- franken_layout_by_page


def test_layout_by_page_from_pdf_schema():
    payload = {
        "pages": [
            {"page": 1, "layout": [{"label": "text", "boxes": [[1, 2, 3, 4]]}]},
            {"page": 2, "layout": []},
        ]
    }
    assert franken_layout_by_page(payload) == {
        "1": [{"label": "text", "boxes": [[1, 2, 3, 4]]}],
        "2": [],
    }


def test_layout_by_page_from_single_image_schema():
    payload = {"layout": [{"label": "figure", "boxes": [[0, 0, 1, 1]]}]}
    assert franken_layout_by_page(payload) == {
        "1": [{"label": "figure", "boxes": [[0, 0, 1, 1]]}]
    }


def test_layout_by_page_missing_keys_is_empty():
    assert franken_layout_by_page({"markdown": "x"}) == {}


# ---------------------------------------------------------------- parse_with_franken


def test_parse_with_franken_metadata_carries_layout(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    payload = {
        "markdown": "Page one.\n<PAGE>\nPage two.",
        "pages": [
            {"page": 1, "layout": [{"label": "text", "boxes": [[1, 2, 3, 4]]}]},
            {"page": 2, "layout": []},
        ],
    }

    calls: list[tuple[Path, bool]] = []

    def fake_runner(p: Path, multi_page: bool) -> dict:
        calls.append((p, multi_page))
        return payload

    ir = parse_with_franken(
        pdf,
        Settings(),
        runner=fake_runner,
        page_count_fn=lambda _p: 2,
    )

    assert calls == [(pdf, True)]  # one subprocess call, multi_page since n_pages > 1
    assert ir.metadata["franken_layout"] == {
        "1": [{"label": "text", "boxes": [[1, 2, 3, 4]]}],
        "2": [],
    }
    assert ir.mime_type == "application/pdf"
    assert ir.source_uri == pdf.resolve().as_uri()
    assert len(ir.blocks) == 2
    assert [b.block_id for b in ir.blocks] == ["#/franken/p1/0", "#/franken/p2/0"]


def test_parse_with_franken_single_page_does_not_pass_multi_page(tmp_path):
    pdf = tmp_path / "one.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    calls: list[tuple[Path, bool]] = []

    def fake_runner(p: Path, multi_page: bool) -> dict:
        calls.append((p, multi_page))
        return {"markdown": "Only page.", "layout": []}

    parse_with_franken(pdf, Settings(), runner=fake_runner, page_count_fn=lambda _p: 1)
    assert calls == [(pdf, False)]


# ---------------------------------------------------------------- default runner (_run_focr)


def test_run_focr_raises_runtime_error_with_stderr_on_nonzero_exit(tmp_path, monkeypatch):
    import subprocess as subprocess_mod

    class FakeCompletedProcess:
        returncode = 1
        stdout = b""
        stderr = b"boom: focr crashed"

    captured_args = {}

    def fake_run(args, capture_output):
        captured_args["args"] = args
        assert capture_output is True
        return FakeCompletedProcess()

    monkeypatch.setattr(subprocess_mod, "run", fake_run)

    pdf = tmp_path / "bad.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    with pytest.raises(RuntimeError, match="boom: focr crashed"):
        _run_focr(pdf, False, Settings(franken_bin="focr"))

    assert captured_args["args"] == ["focr", "ocr", str(pdf), "--json"]


def test_run_focr_appends_multi_page_flag_and_uses_franken_bin(tmp_path, monkeypatch):
    import subprocess as subprocess_mod

    class FakeCompletedProcess:
        returncode = 0
        stdout = b'{"schema_version": 1, "markdown": "hi"}'
        stderr = b""

    captured_args = {}

    def fake_run(args, capture_output):
        captured_args["args"] = args
        return FakeCompletedProcess()

    monkeypatch.setattr(subprocess_mod, "run", fake_run)

    pdf = tmp_path / "multi.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    result = _run_focr(pdf, True, Settings(franken_bin="/opt/focr/focr"))
    assert captured_args["args"] == ["/opt/focr/focr", "ocr", str(pdf), "--json", "--multi-page"]
    assert result == {"schema_version": 1, "markdown": "hi"}


# ---------------------------------------------------------------- router wiring


def _profile(cov: float) -> DocProfile:
    return DocProfile(page_count=10, pages_with_text=int(cov * 10), text_coverage=cov)


def test_route_scanned_with_franken_bin_and_no_vlm_routes_frankenocr():
    s = Settings(franken_bin="focr", vlm_endpoint=None)
    assert route(_profile(0.1), s) == "frankenocr"


def test_route_scanned_without_franken_or_vlm_falls_back_to_paddle():
    s = Settings(franken_bin=None, vlm_endpoint=None)
    assert route(_profile(0.1), s) == "paddleocr"


def test_route_vlm_takes_priority_over_frankenocr():
    s = Settings(franken_bin="focr", vlm_endpoint="http://gpu:10000/v1")
    assert route(_profile(0.1), s) == "vlm"


def test_route_digital_still_routes_docling_regardless_of_franken_bin():
    s = Settings(franken_bin="focr")
    assert route(_profile(0.95), s) == "docling"
