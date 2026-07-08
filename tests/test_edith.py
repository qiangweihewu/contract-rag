from __future__ import annotations

from pathlib import Path

import pytest

from contract_rag.config import Settings
from contract_rag.eval.edith import (
    DocEntry,
    analyze_routing,
    chars_per_page,
    confirm_doc,
    format_parse_checks,
    format_report,
    load_index,
    select_mixed,
    summarize,
)
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR
from contract_rag.parse.probe import PageProfile

_CSV = (
    "doc_id,question_id,role,entity,filename,classification,language,format,pages,description\n"
    # same physical file recurs under a fresh doc_id per use case → must collapse to 1
    "DOC-1a,CEO-01,CEO,veracier_sa,contrats/a.pdf,YES,en,mixed,4,first\n"
    "DOC-1b,LEGAL-02,GC,veracier_sa,contrats/a.pdf,YES,en,mixed,4,dup use case\n"
    "DOC-2,CEO-01,CEO,veracier_uk,juridique/b.pdf,NO,fr,searchable,3,digital\n"
    "DOC-3,QUAL-01,QD,veracier_gmbh,qualite/c.pdf,NCR,de,scanned,2,scan\n"
    "DOC-4,HR-01,CHRO,veracier_aero,rh/d.pdf,SUMMARY,en,mixed,6,another mixed\n"
    # same filename under a different entity is a DIFFERENT physical file → keep both
    "DOC-5,CEO-02,CEO,veracier_gmbh,contrats/a.pdf,YES,de,mixed,4,homonym file\n"
)


def _pp(page: int, has_text: bool) -> PageProfile:
    return PageProfile(page=page, char_count=700 if has_text else 0, has_text=has_text)


# ------------------------------------------------------------------ index loading

def test_load_index_dedupes_by_physical_path(tmp_path: Path):
    (tmp_path / "MASTER_INDEX.csv").write_text(_CSV)
    entries = load_index(tmp_path)
    # DOC-1a/DOC-1b collapse (same path); DOC-5 kept (same filename, other entity)
    assert [e.doc_id for e in entries] == ["DOC-1a", "DOC-2", "DOC-3", "DOC-4", "DOC-5"]
    a = entries[0]
    assert a.rel_path() == "by_entity/veracier_sa/contrats/a.pdf"
    assert a.pages == 4 and a.format == "mixed"


def test_select_mixed_prefers_mixed_sorted_and_capped(tmp_path: Path):
    (tmp_path / "MASTER_INDEX.csv").write_text(_CSV)
    entries = load_index(tmp_path)
    mixed = select_mixed(entries)
    # sorted by rel_path: veracier_aero < veracier_gmbh < veracier_sa
    assert [e.doc_id for e in mixed] == ["DOC-4", "DOC-5", "DOC-1a"]
    assert [e.doc_id for e in select_mixed(entries, cap=1)] == ["DOC-4"]


def test_select_mixed_falls_back_to_scanned_when_no_mixed():
    entries = [
        DocEntry(doc_id="s1", entity="e", filename="x.pdf", language="en",
                 format="scanned", pages=2),
        DocEntry(doc_id="d1", entity="e", filename="y.pdf", language="en",
                 format="searchable", pages=2),
    ]
    assert [e.doc_id for e in select_mixed(entries)] == ["s1"]


# ------------------------------------------------------------- routing analysis

def test_analyze_routing_mixed_balanced_routes_whole_doc_to_paddle():
    # 2 digital + 2 scanned → coverage 0.5 < 0.8 → the WHOLE doc goes to paddle,
    # so the 2 clean digital pages get needlessly re-OCR'd (degraded, not lost).
    pages = [_pp(1, True), _pp(2, True), _pp(3, False), _pp(4, False)]
    r = analyze_routing("bal", pages, Settings())
    assert r.doc_coverage == 0.5 and r.doc_route == "paddleocr"
    assert r.n_digital_pages == 2 and r.n_scanned_pages == 2
    assert r.page_routes == ["docling", "docling", "paddleocr", "paddleocr"]
    assert r.n_misrouted_pages == 2       # the 2 digital pages
    assert r.n_content_loss_pages == 0    # paddle DOES ocr the scanned pages
    assert r.n_degraded_pages == 2
    assert r.n_segments == 2


def test_analyze_routing_mostly_digital_loses_scanned_annex():
    # 4 digital + 1 scanned → coverage 0.8 == threshold → docling for the whole doc,
    # and docling's OCR is off, so the scanned signature page comes back EMPTY: lost.
    pages = [_pp(1, True), _pp(2, True), _pp(3, True), _pp(4, True), _pp(5, False)]
    r = analyze_routing("annex", pages, Settings())
    assert r.doc_coverage == 0.8 and r.doc_route == "docling"
    assert r.n_misrouted_pages == 1
    assert r.n_content_loss_pages == 1    # scanned page → docling → text lost
    assert r.n_degraded_pages == 0


def test_analyze_routing_pure_digital_no_misroute():
    pages = [_pp(1, True), _pp(2, True)]
    r = analyze_routing("pure", pages, Settings())
    assert r.doc_route == "docling" and r.n_misrouted_pages == 0
    assert r.n_content_loss_pages == 0 and r.n_segments == 1


# ---------------------------------------------------------------- aggregation

def test_summarize_counts_misroute_and_loss():
    routings = [
        analyze_routing("bal", [_pp(1, True), _pp(2, False)], Settings()),  # paddle, degrade 1
        analyze_routing(
            "annex", [_pp(i, i < 5) for i in range(1, 6)], Settings()
        ),  # docling, loss 1
        analyze_routing("pure", [_pp(1, True)], Settings()),  # docling, no misroute
    ]
    s = summarize(routings, subset="mixed")
    assert s.n_docs == 3 and s.subset == "mixed"
    assert s.n_docs_both_formats == 2
    assert s.n_docs_with_misroute == 2
    assert s.frac_docs_with_misroute == pytest.approx(2 / 3, abs=1e-3)
    assert s.total_content_loss_pages == 1 and s.n_docs_content_loss == 1
    assert s.total_degraded_pages == 1 and s.n_docs_degraded == 1
    assert s.doc_route_dist == {"paddleocr": 1, "docling": 2}


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        summarize([], subset="mixed")


def test_format_report_smoke():
    r = analyze_routing("d", [_pp(1, True), _pp(2, False)], Settings())
    out = format_report([r], summarize([r], subset="mixed"))
    assert "misrouted" in out.lower() and "d" in out


# ------------------------------------------------------ parse confirmation

def _block(page: int, text: str, engine: str) -> DocBlock:
    return DocBlock(
        block_id=f"#/{engine}/{page}", type=BlockType.PARAGRAPH, text=text,
        bbox=BoundingBox(page=page, x0=0, y0=0, x1=1, y1=1),
        confidence=1.0, source_engine=engine,
    )


def _ir(blocks: list[DocBlock]) -> DocumentIR:
    return DocumentIR(
        doc_id="d", source_uri="file:///d", file_hash="h",
        mime_type="application/pdf", blocks=blocks,
    )


def test_chars_per_page_sums_block_text_by_page():
    ir = _ir([_block(1, "abcd", "docling"), _block(1, "ef", "docling"), _block(3, "xyz", "p")])
    assert chars_per_page(ir, 3) == [6, 0, 3]


def test_confirm_doc_shows_scanned_page_recovery(tmp_path: Path):
    pdf = tmp_path / "m.pdf"
    pages = [_pp(1, True), _pp(2, False)]  # scanned page 2

    # doc-level route (coverage 0.5 → paddle here) but simulate the LOSS case: pretend
    # today's parse recovered text on the digital page only, nothing on the scanned one
    doclevel = _ir([_block(1, "digital body text", "docling")])
    # per-page router recovers the scanned annex via paddle
    perpage = _ir([_block(1, "digital body text", "docling"), _block(2, "SIGNED ANNEX", "paddleocr")])

    check = confirm_doc(
        pdf, pages, Settings(),
        doclevel_parse=lambda _p: doclevel,
        perpage_parse=lambda _p: perpage,
    )
    assert check.scanned_pages == [2]
    assert check.scanned_chars_doclevel == 0
    assert check.scanned_chars_perpage == len("SIGNED ANNEX")
    out = format_parse_checks([check])
    assert "recovered" in out
