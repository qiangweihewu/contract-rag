from contract_rag.clean.quality import QualityReport
from contract_rag.demo.report import (
    FieldRow,
    ReportData,
    build_report_data,
    compare_fields,
    field_status,
    render_html,
)


def _qr(score, needs):
    return QualityReport(quality_score=score, garble_ratio=0.0, empty_ratio=0.0,
                         table_integrity=1.0, mean_confidence=1.0, needs_review=needs)


def _data(fields):
    return ReportData(doc_id="d", dirty_quality=_qr(0.6, True), cleaned_quality=_qr(0.97, False),
                      fields=fields, dirty_sample="", cleaned_sample="")
from contract_rag.extract.rules import RuleExtractor
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _b(text, bid):
    return DocBlock(block_id=bid, type=BlockType.PARAGRAPH, text=text,
                    bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                    confidence=1.0, source_engine="docling")


def _ir(blocks):
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", blocks=blocks, metadata={})


def test_build_report_data_shows_quality_recovery_and_verified_facts():
    ir = _ir([
        _b("This Agreement is entered into by and between Acme Inc. and Globex LLC.", "#/b/0"),
        _b("This Agreement shall be governed by the laws of the State of New York.", "#/b/1"),
        _b("Either party may terminate upon ninety (90) days written notice.", "#/b/2"),
    ])
    data = build_report_data(ir, RuleExtractor(), seed=0)

    # cleaning recovers quality (a tiny fixture won't cross needs_review; real contracts do)
    assert data.cleaned_quality.quality_score > data.dirty_quality.quality_score
    assert data.cleaned_quality.garble_ratio < data.dirty_quality.garble_ratio
    cp = next(f for f in data.fields if f.field == "counterparty")
    assert "Acme" in cp.cleaned_value
    assert cp.verified is True               # cleaned value is attributable to its block


def test_field_status_maps_verified_review_notfound():
    def mk(value, verified):
        return FieldRow(field="x", dirty_value="", cleaned_value=value,
                        source_block_id=None, confidence=0.5, verified=verified, reasons=[])
    assert field_status(mk("Acme Inc.", True)) == "verified"
    assert field_status(mk("Acme Inc.", False)) == "review"
    assert field_status(mk("", False)) == "not found"


def test_compare_fields_pairs_two_backends_by_field():
    def fr(field, val, verified):
        return FieldRow(field=field, dirty_value="", cleaned_value=val,
                        source_block_id="b" if val else None, confidence=0.7, verified=verified, reasons=[])

    a = _data([fr("counterparty", "Acme Inc.", True), fr("total_value", "", False)])
    b = _data([fr("counterparty", "Acme Inc.; Globex LLC", True), fr("total_value", "$5,000", False)])
    rows = compare_fields(a, b)
    cp = next(r for r in rows if r["field"] == "counterparty")
    assert cp["a_value"] == "Acme Inc." and cp["a_status"] == "verified"
    assert cp["b_value"] == "Acme Inc.; Globex LLC" and cp["b_status"] == "verified"
    tv = next(r for r in rows if r["field"] == "total_value")
    assert tv["a_status"] == "not found" and tv["b_status"] == "review"


def test_render_html_is_self_contained_and_shows_the_story():
    data = ReportData(
        doc_id="ACME_MSA",
        dirty_quality=QualityReport(quality_score=0.62, garble_ratio=0.70, empty_ratio=0.20,
                                    table_integrity=1.0, mean_confidence=1.0, needs_review=True),
        cleaned_quality=QualityReport(quality_score=0.97, garble_ratio=0.0, empty_ratio=0.0,
                                      table_integrity=1.0, mean_confidence=1.0, needs_review=False),
        fields=[FieldRow(field="counterparty", dirty_value="", cleaned_value="Acme Inc.; Globex LLC",
                         source_block_id="#/b/0", confidence=0.5, verified=True, reasons=[])],
        dirty_sample="TheÂ partiesÂ agree",
        cleaned_sample="The parties agree",
    )
    html = render_html(data)
    assert html.lstrip().lower().startswith("<!doctype html")
    assert "ACME_MSA" in html
    assert "0.62" in html and "0.97" in html          # before/after scores
    assert "counterparty" in html
    assert "Acme Inc.; Globex LLC" in html            # the recovered, verified value
    assert "<style" in html                           # self-contained (no external CSS file)
