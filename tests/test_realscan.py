from __future__ import annotations

from pathlib import Path

import pytest

from contract_rag.config import Settings
from contract_rag.eval.realscan import (
    DocQuality,
    Zone,
    block_overlaps,
    evaluate_doc,
    format_report,
    list_input_docs,
    overlap_stats,
    parse_gedi,
    percentile,
    scale_zones,
    summarize,
    zone_scale,
)
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR
from contract_rag.parse.probe import DocProfile


def _block(text: str, conf: float = 0.9, bbox: tuple | None = None) -> DocBlock:
    return DocBlock(
        block_id=f"#/b/{text[:8]}",
        type=BlockType.PARAGRAPH,
        text=text,
        bbox=BoundingBox(page=1, x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3])
        if bbox
        else None,
        confidence=conf,
        source_engine="test",
    )


def _ir(blocks: list[DocBlock]) -> DocumentIR:
    return DocumentIR(
        doc_id="d", source_uri="file:///d", file_hash="h", mime_type="application/pdf",
        blocks=blocks,
    )


# ---------------------------------------------------------------- input listing

def test_list_input_docs_sorted_and_capped(tmp_path: Path):
    for name in ["c.tif", "a.pdf", "b.png", "d.txt", "e.TIFF"]:
        (tmp_path / name).write_bytes(b"x")
    docs = list_input_docs(tmp_path, cap=100)
    assert [d.name for d in docs] == ["a.pdf", "b.png", "c.tif", "e.TIFF"]
    assert [d.name for d in list_input_docs(tmp_path, cap=2)] == ["a.pdf", "b.png"]


def test_image_to_pdf_roundtrip(tmp_path: Path):
    PIL = pytest.importorskip("PIL")
    pdfium = pytest.importorskip("pypdfium2")
    from PIL import Image

    from contract_rag.eval.realscan import image_to_pdf

    img = tmp_path / "scan.tif"
    Image.new("1", (300, 450), 1).save(img, dpi=(150, 150))
    out = image_to_pdf(img, tmp_path / "scan.pdf")
    pdf = pdfium.PdfDocument(str(out))
    try:
        assert len(pdf) == 1
        w, h = pdf[0].get_size()
        # 300px @150dpi = 2in = 144pt
        assert w == pytest.approx(144, abs=1)
        assert h == pytest.approx(216, abs=1)
    finally:
        pdf.close()


# ---------------------------------------------------------------- evaluate_doc

def test_evaluate_doc_routes_scores_and_cleans():
    # real utf-8→latin-1 mojibake (what dirtify injects and ftfy repairs)
    moji = "Agreement with café counterparty".encode("utf-8").decode("latin-1")
    garbled = [_block(moji), _block("clean text", conf=0.5)]

    def fake_probe(_p):
        return DocProfile(page_count=1, pages_with_text=0, text_coverage=0.0)

    calls: list[str] = []

    def fake_paddle(_p, _s):
        calls.append("paddleocr")
        return _ir(garbled)

    dq = evaluate_doc(
        Path("x.pdf"),
        Settings(),
        probe_fn=fake_probe,
        adapters={"paddleocr": fake_paddle},
        name="x",
    )
    assert calls == ["paddleocr"]  # coverage 0.0 + no VLM endpoint → paddle branch
    assert dq.engine == "paddleocr"
    assert dq.text_coverage == 0.0
    assert dq.raw.garble_ratio == 0.5
    # fix_unicode repairs the mojibake block → cleaned garble drops, score rises
    assert dq.cleaned.garble_ratio == 0.0
    assert dq.cleaned.quality_score > dq.raw.quality_score


def test_evaluate_doc_high_coverage_routes_to_docling():
    def fake_probe(_p):
        return DocProfile(page_count=2, pages_with_text=2, text_coverage=1.0)

    dq = evaluate_doc(
        Path("x.pdf"),
        Settings(),
        probe_fn=fake_probe,
        adapters={"docling": lambda _p, _s: _ir([_block("hello")])},
    )
    assert dq.engine == "docling"
    assert dq.page_count == 2


# ---------------------------------------------------------------- aggregation

def test_percentile_nearest_rank():
    vals = [float(i) for i in range(1, 11)]
    assert percentile(vals, 0.1) == 1.0
    assert percentile(vals, 0.5) == 5.0
    assert percentile([3.0], 0.1) == 3.0
    with pytest.raises(ValueError):
        percentile([], 0.5)


def _dq(name: str, raw_q: float, clean_q: float, engine: str = "paddleocr") -> DocQuality:
    from contract_rag.clean.quality import QualityReport

    def rep(q: float) -> QualityReport:
        return QualityReport(
            quality_score=q, garble_ratio=0.1, empty_ratio=0.0, table_integrity=1.0,
            mean_confidence=0.8, needs_review=q < 0.75,
        )

    return DocQuality(
        name=name, engine=engine, page_count=1, text_coverage=0.0,
        raw=rep(raw_q), cleaned=rep(clean_q),
    )


def test_summarize_aggregates_and_review_rate():
    results = [_dq("a", 0.5, 0.8), _dq("b", 0.7, 0.9), _dq("c", 0.9, 0.95, engine="docling")]
    s = summarize(results)
    assert s.n_docs == 3 and s.total_pages == 3
    assert s.engines == {"paddleocr": 2, "docling": 1}
    assert s.raw.mean_quality == 0.7
    assert s.raw.needs_review_rate == pytest.approx(2 / 3, abs=1e-3)
    assert s.cleaned.needs_review_rate == 0.0
    assert s.raw.p10_quality == 0.5


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        summarize([])


def test_format_report_contains_docs_and_aggregate():
    results = [_dq("mydoc", 0.5, 0.8)]
    out = format_report(results, summarize(results))
    assert "mydoc" in out and "cleaned" in out and "review" in out


# ------------------------------------------------- GEDI groundtruth + occlusion

_GEDI = """<?xml version="1.0" encoding="UTF-8"?>
<GEDI xmlns="http://lamp.cfar.umd.edu/GEDI" version="1.0">
  <DL_DOCUMENT src="x.tif" NrOfPages="1" docTag="xml">
    <DL_PAGE gedi_type="DL_PAGE" src="x.tif" pageID="1" width="1200" height="1575">
      <DL_ZONE gedi_type="DLSignature" col="651" row="1123" width="200" height="72"> </DL_ZONE>
      <DL_ZONE gedi_type="DLLogo" col="134" row="150" width="122" height="120"> </DL_ZONE>
    </DL_PAGE>
  </DL_DOCUMENT>
</GEDI>"""


def test_parse_gedi_extracts_page_and_zones():
    pz = parse_gedi(_GEDI)
    assert (pz.width, pz.height) == (1200.0, 1575.0)
    assert [z.kind for z in pz.zones] == ["DLSignature", "DLLogo"]
    sig = pz.zones[0]
    assert (sig.x0, sig.y0, sig.x1, sig.y1) == (651.0, 1123.0, 851.0, 1195.0)


def test_zone_scale_and_scaling():
    assert zone_scale(150.0) == 2.0  # 150dpi image rendered at 300dpi → 2x
    z = Zone(kind="DLSignature", x0=10, y0=20, x1=30, y1=40)
    (s,) = scale_zones([z], 2.0)
    assert (s.x0, s.y0, s.x1, s.y1) == (20.0, 40.0, 60.0, 80.0)


def test_block_overlaps_intersection_only():
    zones = [Zone(kind="DLSignature", x0=100, y0=100, x1=200, y1=200)]
    assert block_overlaps(_block("in", bbox=(150, 150, 250, 250)), zones)
    assert not block_overlaps(_block("out", bbox=(300, 300, 400, 400)), zones)
    assert not block_overlaps(_block("edge", bbox=(200, 100, 300, 200)), zones)  # touching
    assert not block_overlaps(_block("nobox"), zones)


def test_overlap_stats_pools_conf_and_garble():
    zones = [Zone(kind="DLSignature", x0=0, y0=0, x1=100, y1=100)]
    blocks = [
        _block("Ã¢â€ garbled", conf=0.3, bbox=(10, 10, 50, 50)),   # overlap, garbled
        _block("under sig", conf=0.5, bbox=(60, 60, 90, 90)),      # overlap, clean
        _block("clean body text", conf=0.95, bbox=(200, 200, 300, 300)),
    ]
    st = overlap_stats([(blocks, zones)])
    assert (st.n_pages, st.n_overlap, st.n_other) == (1, 2, 1)
    assert st.mean_conf_overlap == 0.4
    assert st.mean_conf_other == 0.95
    assert st.garble_rate_overlap == 0.5
    assert st.garble_rate_other == 0.0


def test_overlap_stats_empty_sides_are_none():
    st = overlap_stats([([], [])])
    assert st.mean_conf_overlap is None and st.garble_rate_other is None
