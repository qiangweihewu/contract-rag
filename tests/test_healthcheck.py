"""One-command PoC health-check pack (`contract_rag.healthcheck`). Dep-free: fake
parse/extract seams, hand-built IRs — the codebase's usual DI style (see
test_demo_batch.py / test_export.py, which this mirrors)."""
from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path

import pytest

from contract_rag.demo.export import COLUMNS
from contract_rag.extract.rules import RuleExtractor
from contract_rag.healthcheck.core import (
    DocFailure,
    DocOutcome,
    DocTimeoutError,
    build_summary,
    build_summary_html,
    default_parse_fn,
    discover_docs,
    process_document,
    run_healthcheck,
    run_with_timeout,
)
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _block(text: str, bid: str, source_engine: str = "docling", confidence: float = 1.0) -> DocBlock:
    return DocBlock(
        block_id=bid, type=BlockType.PARAGRAPH, text=text,
        bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
        confidence=confidence, source_engine=source_engine,
    )


def _digital_ir(doc_id: str = "d") -> DocumentIR:
    return DocumentIR(
        doc_id=doc_id, source_uri=f"file:///{doc_id}", file_hash="h",
        mime_type="application/pdf",
        blocks=[
            _block("This Agreement is by and between Acme Inc. and Globex LLC.", "#/b/0"),
            _block("Governed by the laws of the State of New York.", "#/b/1"),
        ],
        metadata={},
    )


def _scanned_signed_ir(doc_id: str = "s") -> DocumentIR:
    return DocumentIR(
        doc_id=doc_id, source_uri=f"file:///{doc_id}", file_hash="h",
        mime_type="application/pdf",
        blocks=[
            _block("This Agreement is by and between Acme Inc. and Globex LLC.", "#/b/0",
                   source_engine="paddleocr"),
            _block("Governed by the laws of the State of New York.", "#/b/1",
                   source_engine="paddleocr"),
            _block("Very truly yours,", "#/b/2", source_engine="paddleocr"),
        ],
        metadata={},
    )


def _scanned_unsigned_ir(doc_id: str = "u") -> DocumentIR:
    return DocumentIR(
        doc_id=doc_id, source_uri=f"file:///{doc_id}", file_hash="h",
        mime_type="application/pdf",
        blocks=[
            _block("This Agreement is by and between Acme Inc. and Globex LLC.", "#/b/0",
                   source_engine="paddleocr"),
        ],
        metadata={},
    )


# ---------------------------------------------------------------- discovery


def test_discover_docs_finds_only_pdf_and_docx(tmp_path: Path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "b.docx").write_bytes(b"PK")
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / ".DS_Store").write_bytes(b"")
    found = discover_docs(tmp_path)
    assert [p.name for p in found] == ["a.pdf", "b.docx"]


def test_discover_docs_is_case_insensitive_on_suffix(tmp_path: Path):
    (tmp_path / "A.PDF").write_bytes(b"%PDF")
    found = discover_docs(tmp_path)
    assert [p.name for p in found] == ["A.PDF"]


def test_discover_docs_on_nonexistent_dir_raises():
    with pytest.raises(ValueError):
        discover_docs(Path("/no/such/dir/at/all"))


# ---------------------------------------------------------------- default_parse_fn dispatch


def test_default_parse_fn_routes_docx_straight_to_docling(monkeypatch, tmp_path: Path):
    calls: list[Path] = []

    def fake_docling(path: Path) -> DocumentIR:
        calls.append(Path(path))
        return _digital_ir()

    monkeypatch.setattr("contract_rag.parse.docling_parser.parse_with_docling", fake_docling)
    p = tmp_path / "contract.docx"
    p.write_bytes(b"PK")
    ir = default_parse_fn(p, settings=object())
    assert calls == [p]
    assert ir.doc_id == "d"


def test_default_parse_fn_routes_pdf_through_the_router(monkeypatch, tmp_path: Path):
    calls: list[tuple[Path, object]] = []

    def fake_route(path: Path, settings) -> DocumentIR:
        calls.append((Path(path), settings))
        return _digital_ir()

    monkeypatch.setattr("contract_rag.parse.router.parse", fake_route)
    p = tmp_path / "contract.pdf"
    p.write_bytes(b"%PDF")
    settings = object()
    ir = default_parse_fn(p, settings=settings)
    assert calls == [(p, settings)]
    assert ir.doc_id == "d"


# ---------------------------------------------------------------- process_document


def test_process_document_digital_happy_path(tmp_path: Path):
    p = tmp_path / "acme.pdf"
    p.write_bytes(b"%PDF")
    outcome = process_document(p, RuleExtractor(), parse_fn=lambda _p: _digital_ir())
    assert isinstance(outcome, DocOutcome)
    assert outcome.filename == "acme.pdf"
    assert outcome.source_engine == "digital"
    assert 0.0 <= outcome.quality_score <= 1.0
    assert outcome.signature is None            # signature audit only runs on scanned docs
    assert outcome.stp["total_fields"] > 0
    assert len(outcome.facts_rows) == outcome.stp["total_fields"]
    for row in outcome.facts_rows:
        assert set(row) == set(COLUMNS)
    assert outcome.report_html.lstrip().lower().startswith("<!doctype html")


def test_process_document_scanned_runs_signature_audit_and_signed_fires(tmp_path: Path):
    p = tmp_path / "letter.pdf"
    p.write_bytes(b"%PDF")
    outcome = process_document(p, RuleExtractor(), parse_fn=lambda _p: _scanned_signed_ir())
    assert outcome.source_engine == "scanned"
    assert outcome.signature is not None
    assert outcome.signature["signed"] is True
    assert "closing" in outcome.signature["signals"]


def test_process_document_scanned_unsigned_does_not_fire_signature(tmp_path: Path):
    p = tmp_path / "memo.pdf"
    p.write_bytes(b"%PDF")
    outcome = process_document(p, RuleExtractor(), parse_fn=lambda _p: _scanned_unsigned_ir())
    assert outcome.signature is not None
    assert outcome.signature["signed"] is False


def test_process_document_timeout_raises(tmp_path: Path):
    p = tmp_path / "slow.pdf"
    p.write_bytes(b"%PDF")

    def slow_parse(_p: Path) -> DocumentIR:
        time.sleep(0.2)
        return _digital_ir()

    with pytest.raises(DocTimeoutError):
        process_document(p, RuleExtractor(), parse_fn=slow_parse, timeout=0.02)


def test_run_with_timeout_returns_result_when_fast_enough():
    assert run_with_timeout(lambda: 42, timeout=5.0) == 42


# ---------------------------------------------------------------- run_healthcheck


def _write_pdf(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.4 fake")
    return p


def test_run_healthcheck_happy_path_writes_pack_and_aggregates(tmp_path: Path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    out_dir = tmp_path / "out"
    a = _write_pdf(input_dir, "a.pdf")
    b = _write_pdf(input_dir, "b.pdf")

    def parse_fn(path: Path) -> DocumentIR:
        return _digital_ir() if path == a else _scanned_signed_ir()

    summary = run_healthcheck(input_dir, out_dir, RuleExtractor(), parse_fn=parse_fn)

    assert summary.n_docs == 2
    assert summary.n_ok == 2
    assert summary.n_failed == 0
    assert summary.engine_counts == {"digital": 1, "scanned": 1}
    assert summary.signature_counts == {"signed": 1, "unsigned": 0}
    assert summary.quality_mean is not None and summary.quality_min is not None
    assert summary.quality_min <= summary.quality_mean

    # per-doc reports
    assert (out_dir / "a.html").exists()
    assert (out_dir / "b.html").exists()
    # combined facts export
    assert (out_dir / "facts.csv").exists()
    assert (out_dir / "facts.json").exists()
    combined = json.loads((out_dir / "facts.json").read_text())
    assert set(combined) == {"facts", "stp"}
    assert {r["doc_id"] for r in combined["facts"]}  # non-empty
    csv_rows = list(csv.DictReader(io.StringIO((out_dir / "facts.csv").read_text())))
    assert csv_rows and set(csv_rows[0]) == set(COLUMNS)
    # summary pack
    assert (out_dir / "summary.html").exists()
    assert (out_dir / "summary.json").exists()
    html = (out_dir / "summary.html").read_text()
    assert "a.html" in html and "b.html" in html


def test_run_healthcheck_one_doc_fails_batch_continues(tmp_path: Path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    out_dir = tmp_path / "out"
    good = _write_pdf(input_dir, "good.pdf")
    bad = _write_pdf(input_dir, "corrupt.pdf")

    def parse_fn(path: Path) -> DocumentIR:
        if path == bad:
            raise ValueError("could not parse: encrypted or corrupt PDF")
        return _digital_ir()

    summary = run_healthcheck(input_dir, out_dir, RuleExtractor(), parse_fn=parse_fn)

    assert summary.n_docs == 2
    assert summary.n_ok == 1
    assert summary.n_failed == 1
    assert len(summary.failures) == 1
    assert summary.failures[0] == DocFailure(
        filename="corrupt.pdf", reason="ValueError: could not parse: encrypted or corrupt PDF"
    )
    # the good doc still got its report + is reflected in aggregates
    assert (out_dir / "good.html").exists()
    assert not (out_dir / "corrupt.html").exists()
    assert summary.quality_mean is not None
    html = (out_dir / "summary.html").read_text()
    assert "corrupt.pdf" in html   # failed-docs table


def test_run_healthcheck_all_docs_fail_still_writes_a_pack(tmp_path: Path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    out_dir = tmp_path / "out"
    _write_pdf(input_dir, "bad.pdf")

    def parse_fn(_path: Path) -> DocumentIR:
        raise ValueError("zero-page PDF")

    summary = run_healthcheck(input_dir, out_dir, RuleExtractor(), parse_fn=parse_fn)
    assert summary.n_ok == 0
    assert summary.n_failed == 1
    assert summary.quality_mean is None
    assert (out_dir / "summary.html").exists()
    assert (out_dir / "facts.csv").exists()  # empty pack, still a valid (empty) file


def test_run_healthcheck_timeout_records_a_failure_not_a_crash(tmp_path: Path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    out_dir = tmp_path / "out"
    _write_pdf(input_dir, "slow.pdf")

    def slow_parse(_path: Path) -> DocumentIR:
        time.sleep(0.2)
        return _digital_ir()

    summary = run_healthcheck(input_dir, out_dir, RuleExtractor(), parse_fn=slow_parse, timeout=0.02)
    assert summary.n_failed == 1
    assert "DocTimeoutError" in summary.failures[0].reason


def test_run_healthcheck_empty_input_dir_raises_clean_error(tmp_path: Path):
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    with pytest.raises(ValueError, match="no PDF/DOCX"):
        run_healthcheck(input_dir, tmp_path / "out", RuleExtractor(), parse_fn=lambda p: _digital_ir())


def test_run_healthcheck_clm_mapping_is_applied(tmp_path: Path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    out_dir = tmp_path / "out"
    _write_pdf(input_dir, "a.pdf")

    run_healthcheck(input_dir, out_dir, RuleExtractor(), parse_fn=lambda _p: _digital_ir(),
                    clm="salesforce")
    csv_rows = list(csv.DictReader(io.StringIO((out_dir / "facts.csv").read_text())))
    assert any(r["clm_field"] == "GoverningLaw__c" for r in csv_rows)


# ---------------------------------------------------------------- build_summary math


def _outcome(filename: str, engine: str, quality: float, needs_review: bool,
            stp_fields: int, total_fields: int, review_fields: list[str],
            signed: bool | None) -> DocOutcome:
    rows = [
        {"doc_id": filename, "field": f"f{i}", "clm_field": f"f{i}", "value": "x",
         "source_block_id": "#/b/0", "confidence": 0.9, "risk_tier": "medium",
         "verified": i < stp_fields}
        for i in range(total_fields)
    ]
    return DocOutcome(
        filename=filename, doc_id=filename, source_engine=engine,
        quality_score=quality, needs_review=needs_review,
        stp={"stp_fields": stp_fields, "total_fields": total_fields,
             "stp_rate": stp_fields / total_fields if total_fields else 0.0,
             "straight_through": not review_fields, "review_fields": review_fields},
        signature=(None if signed is None else {"signed": signed, "confidence": 0.9,
                                                 "evidence_block_ids": [], "signals": []}),
        facts_rows=rows,
        report_html="<!doctype html><html></html>",
        report_file=f"{filename}.html",
    )


def test_build_summary_aggregates_engine_and_quality():
    outcomes = [
        _outcome("a", "digital", 0.9, False, 2, 2, [], None),
        _outcome("b", "scanned", 0.5, True, 1, 2, ["f1"], True),
        _outcome("c", "scanned", 0.7, False, 2, 2, [], False),
    ]
    summary = build_summary(outcomes, failures=[])
    assert summary.n_docs == 3
    assert summary.n_ok == 3
    assert summary.engine_counts == {"digital": 1, "scanned": 2}
    assert summary.quality_mean == pytest.approx((0.9 + 0.5 + 0.7) / 3)
    assert summary.quality_min == 0.5
    assert summary.needs_review_count == 1
    assert summary.signature_counts == {"signed": 1, "unsigned": 1}


def test_build_summary_stp_rollup_matches_batch_semantics():
    outcomes = [
        _outcome("a", "digital", 0.9, False, 2, 2, [], None),      # 2/2 verified
        _outcome("b", "digital", 0.8, False, 1, 2, ["f1"], None),  # 1/2 verified
    ]
    summary = build_summary(outcomes, failures=[])
    assert summary.stp == {
        "stp_fields": 3, "total_fields": 4, "stp_rate": 0.75,
        "straight_through": False, "review_fields": ["f1"],
    }


def test_build_summary_field_verified_and_quarantined_counts():
    outcomes = [
        _outcome("a", "digital", 0.9, False, 1, 2, ["f1"], None),  # f0 verified, f1 not
        _outcome("b", "digital", 0.8, False, 2, 2, [], None),      # f0,f1 both verified
    ]
    summary = build_summary(outcomes, failures=[])
    assert summary.field_verified_counts == {"f0": 2, "f1": 1}
    assert summary.field_quarantined_counts == {"f1": 1}


def test_build_summary_with_failures_counted_but_not_averaged():
    outcomes = [_outcome("a", "digital", 0.9, False, 1, 1, [], None)]
    failures = [DocFailure(filename="bad.pdf", reason="boom")]
    summary = build_summary(outcomes, failures)
    assert summary.n_docs == 2
    assert summary.n_ok == 1
    assert summary.n_failed == 1
    assert summary.quality_mean == 0.9  # failure excluded, not zero-filled


def test_build_summary_html_renders_docs_and_failures():
    outcomes = [_outcome("a", "digital", 0.9, False, 1, 1, [], None)]
    failures = [DocFailure(filename="bad.pdf", reason="ValueError: nope")]
    summary = build_summary(outcomes, failures)
    html = build_summary_html(summary)
    assert html.lstrip().lower().startswith("<!doctype html")
    assert "a.html" in html
    assert "bad.pdf" in html and "ValueError: nope" in html


# ---------------------------------------------------------------- gated integration


FIXTURE = Path(__file__).parent / "fixtures" / "sample_contract.pdf"


@pytest.mark.skipif(not FIXTURE.exists(), reason="sample_contract.pdf fixture not present")
def test_healthcheck_cli_runs_end_to_end_on_a_real_pdf(tmp_path: Path, monkeypatch):
    import shutil
    import sys

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    shutil.copy(FIXTURE, input_dir / "sample_contract.pdf")
    out_dir = tmp_path / "out"

    monkeypatch.setenv("EXTRACT_BACKEND", "rule")
    monkeypatch.setattr(sys, "argv", ["healthcheck", str(input_dir), str(out_dir)])

    from contract_rag.healthcheck.__main__ import main

    main()

    assert (out_dir / "summary.html").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "facts.csv").exists()
    assert (out_dir / "sample_contract.html").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["n_ok"] == 1
    assert summary["n_failed"] == 0
