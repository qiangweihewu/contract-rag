from pathlib import Path

from contract_rag.demo.batch import build_index_html, run_batch
from contract_rag.eval.golden import GoldenDoc
from contract_rag.extract.rules import RuleExtractor
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _ir():
    b = lambda t, i: DocBlock(block_id=i, type=BlockType.PARAGRAPH, text=t,
                              bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                              confidence=1.0, source_engine="docling")
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h", mime_type="application/pdf",
                      blocks=[b("This Agreement is by and between Acme Inc. and Globex LLC.", "#/b/0"),
                              b("Governed by the laws of the State of New York.", "#/b/1")],
                      metadata={})


def test_build_index_html_lists_docs_with_scores_and_links():
    entries = [{"doc_id": "ACME_MSA", "file": "ACME_MSA.report.html", "dirty": 0.58, "cleaned": 0.97}]
    html = build_index_html(entries)
    assert html.lstrip().lower().startswith("<!doctype html")
    assert "ACME_MSA" in html
    assert "ACME_MSA.report.html" in html       # clickable link
    assert "0.58" in html and "0.97" in html


def test_build_index_html_stp_column_and_aggregates():
    entries = [
        {"doc_id": "a", "file": "a.html", "dirty": 0.5, "cleaned": 0.9,
         "stp_rate": 1.0, "straight_through": True},
        {"doc_id": "b", "file": "b.html", "dirty": 0.5, "cleaned": 0.9,
         "stp_rate": 0.5, "straight_through": False},
    ]
    html = build_index_html(entries)
    assert "100%" in html and "50%" in html               # per-doc STP cell
    assert "Mean field-STP rate 75%" in html               # (1.0 + 0.5) / 2
    assert "50% of docs fully straight-through" in html    # 1 of 2 docs


def test_build_index_html_stp_defaults_when_entries_predate_the_rollup():
    # entries without stp_rate/straight_through (e.g. hand-built, or pre-STP callers)
    # must not crash — they default to 0.0 / not-straight-through.
    entries = [{"doc_id": "ACME_MSA", "file": "ACME_MSA.report.html", "dirty": 0.58, "cleaned": 0.97}]
    html = build_index_html(entries)
    assert "0%" in html


def test_run_batch_writes_a_report_per_doc_plus_index(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    golden = []
    for i in range(2):
        (data_dir / f"doc{i}.pdf").write_bytes(b"%PDF-1.4 fake")
        golden.append(GoldenDoc(doc_id=f"doc{i}", source_pdf=f"doc{i}.pdf", facts={}))
    out = tmp_path / "reports"

    entries = run_batch(golden, data_dir, out, RuleExtractor(), parse_fn=lambda _p: _ir(), seed=0)

    assert len(entries) == 2
    assert (out / "index.html").exists()
    assert (out / "doc0.report.html").exists() and (out / "doc1.report.html").exists()
    assert all(e["cleaned"] >= e["dirty"] for e in entries)


def test_run_batch_entries_carry_stp_rollup(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    golden = []
    for i in range(2):
        (data_dir / f"doc{i}.pdf").write_bytes(b"%PDF-1.4 fake")
        golden.append(GoldenDoc(doc_id=f"doc{i}", source_pdf=f"doc{i}.pdf", facts={}))
    out = tmp_path / "reports"

    entries = run_batch(golden, data_dir, out, RuleExtractor(), parse_fn=lambda _p: _ir(), seed=0)

    assert len(entries) == 2
    for e in entries:
        assert 0.0 <= e["stp_rate"] <= 1.0
        assert isinstance(e["straight_through"], bool)
    index_html = (out / "index.html").read_text()
    assert "Mean field-STP rate" in index_html
    assert "of docs fully straight-through" in index_html


def test_run_batch_json_export_carries_per_doc_and_combined_stp(tmp_path: Path):
    import json

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    golden = []
    for i in range(2):
        (data_dir / f"doc{i}.pdf").write_bytes(b"%PDF-1.4 fake")
        golden.append(GoldenDoc(doc_id=f"doc{i}", source_pdf=f"doc{i}.pdf", facts={}))
    out = tmp_path / "reports"

    run_batch(golden, data_dir, out, RuleExtractor(), parse_fn=lambda _p: _ir(), seed=0,
              export="json", clm="generic")

    per_doc = json.loads((out / "doc0.facts.json").read_text())
    assert set(per_doc) == {"facts", "stp"}
    assert set(per_doc["stp"]) == {"stp_fields", "total_fields", "stp_rate",
                                    "straight_through", "review_fields"}

    combined = json.loads((out / "facts.json").read_text())
    assert set(combined) == {"facts", "stp"}
    assert combined["stp"]["total_fields"] == 2 * per_doc["stp"]["total_fields"]


def test_run_batch_skips_missing_pdfs(tmp_path: Path):
    golden = [GoldenDoc(doc_id="missing", source_pdf="nope.pdf", facts={})]
    entries = run_batch(golden, tmp_path / "data", tmp_path / "out", RuleExtractor(),
                        parse_fn=lambda _p: _ir(), seed=0)
    assert entries == []
