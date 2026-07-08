"""Red/yellow/green status lights + risk-tier grouping for the customer-facing report.

The light is derived only from what exists at customer time (no gold), reusing
verify() semantics: green = passed (attributed + confident), yellow = quarantined
for low confidence (HITL), red = unattributed or empty on a high-risk field."""
from __future__ import annotations

from contract_rag.clean.quality import QualityReport
from contract_rag.demo.report import (
    FieldRow,
    ReportData,
    build_report_data,
    field_status,
    fields_by_tier,
    render_html,
    status_light,
    stp_summary,
)
from contract_rag.extract.rules import RuleExtractor
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _fr(field="x", value="v", verified=True, reasons=(), risk="medium", conf=0.9):
    return FieldRow(field=field, dirty_value="", cleaned_value=value,
                    source_block_id="b" if value else None, confidence=conf,
                    verified=verified, reasons=list(reasons), risk=risk)


# ---------------------------------------------------------------- status_light

def test_green_verified_and_confident():
    assert status_light(_fr(verified=True, reasons=[])) == "green"


def test_yellow_low_confidence_goes_to_hitl():
    assert status_light(_fr(verified=False, reasons=["low_confidence"], conf=0.3)) == "yellow"


def test_red_unattributed_even_if_confident():
    assert status_light(_fr(verified=False, reasons=["unattributed"], conf=0.9)) == "red"
    # unattributed AND low confidence is still red — failed verification dominates
    assert status_light(_fr(verified=False, reasons=["unattributed", "low_confidence"])) == "red"


def test_red_empty_on_high_risk_field():
    assert status_light(_fr(value="", verified=False, reasons=["empty"], risk="high")) == "red"


def test_none_empty_on_lower_risk_field():
    assert status_light(_fr(value="", verified=False, reasons=["empty"], risk="medium")) == "none"
    assert status_light(_fr(value="", verified=False, reasons=["empty"], risk="low")) == "none"


def test_field_status_backcompat_unchanged():
    assert field_status(_fr(verified=True)) == "verified"
    assert field_status(_fr(verified=False)) == "review"
    assert field_status(_fr(value="", verified=False)) == "not found"


# ---------------------------------------------------------------- stp_summary


def test_stp_summary_all_green_is_straight_through():
    fields = [_fr(field="a", verified=True), _fr(field="b", verified=True)]
    s = stp_summary(fields)
    assert s == {
        "stp_fields": 2, "total_fields": 2, "stp_rate": 1.0,
        "straight_through": True, "review_fields": [],
    }


def test_stp_summary_mixed_lists_review_fields_in_order():
    fields = [
        _fr(field="counterparty", verified=True),
        _fr(field="total_value", value="", verified=False, reasons=["empty"], risk="high"),
        _fr(field="governing_law", verified=False, reasons=["low_confidence"], conf=0.3),
        _fr(field="effective_date", verified=True),
    ]
    s = stp_summary(fields)
    assert s["stp_fields"] == 2
    assert s["total_fields"] == 4
    assert s["stp_rate"] == 0.5
    assert s["straight_through"] is False
    assert s["review_fields"] == ["total_value", "governing_law"]  # yellow+red, field order


def test_stp_summary_all_red_is_zero_stp_and_not_straight_through():
    fields = [
        _fr(field="a", verified=False, reasons=["unattributed"]),
        _fr(field="b", value="", verified=False, reasons=["empty"], risk="high"),
    ]
    s = stp_summary(fields)
    assert s["stp_fields"] == 0
    assert s["stp_rate"] == 0.0
    assert s["straight_through"] is False
    assert s["review_fields"] == ["a", "b"]


def test_stp_summary_none_status_does_not_block_straight_through():
    # empty on a lower-risk field is "none" — absence, not a review flag.
    fields = [_fr(field="a", verified=True), _fr(field="b", value="", verified=False,
                                                  reasons=["empty"], risk="medium")]
    s = stp_summary(fields)
    assert s["stp_fields"] == 1
    assert s["total_fields"] == 2
    assert s["straight_through"] is True     # no yellow/red — "none" doesn't count
    assert s["review_fields"] == []


def test_stp_summary_zero_fields_edge_case():
    s = stp_summary([])
    assert s == {
        "stp_fields": 0, "total_fields": 0, "stp_rate": 0.0,
        "straight_through": True, "review_fields": [],
    }


# ---------------------------------------------------------------- tier grouping

def test_fields_by_tier_orders_high_medium_low_and_skips_empty():
    rows = [_fr(field="a", risk="low"), _fr(field="b", risk="high"), _fr(field="c", risk="high")]
    groups = fields_by_tier(rows)
    assert [t for t, _ in groups] == ["high", "low"]          # no medium fields → skipped
    assert [f.field for f in dict(groups)["high"]] == ["b", "c"]


def test_fields_by_tier_unknown_risk_falls_into_medium():
    groups = dict(fields_by_tier([_fr(field="a", risk="medium"), _fr(field="weird", risk="??")]))
    assert {f.field for f in groups["medium"]} == {"a", "weird"}


# ------------------------------------------------- build_report_data + render

def _ir():
    def b(text, bid):
        return DocBlock(block_id=bid, type=BlockType.PARAGRAPH, text=text,
                        bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                        confidence=1.0, source_engine="docling")
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", metadata={}, blocks=[
        b("This Agreement is entered into by and between Acme Inc. and Globex LLC.", "#/b/0"),
        b("This Agreement shall be governed by the laws of the State of New York.", "#/b/1"),
    ])


def test_build_report_data_stamps_risk_from_active_vertical():
    data = build_report_data(_ir(), RuleExtractor(), seed=0)
    risk = {f.field: f.risk for f in data.fields}
    assert risk["total_value"] == "high"
    assert risk["counterparty"] == "medium"
    assert risk["effective_date"] == "low"


def test_render_html_groups_by_tier_with_status_lights():
    q = QualityReport(quality_score=0.9, garble_ratio=0.0, empty_ratio=0.0,
                      table_integrity=1.0, mean_confidence=1.0, needs_review=False)
    data = ReportData(
        doc_id="d", dirty_quality=q, cleaned_quality=q, dirty_sample="", cleaned_sample="",
        fields=[
            _fr(field="total_value", value="", verified=False, reasons=["empty"], risk="high"),
            _fr(field="counterparty", value="Acme Inc.", verified=True, risk="medium"),
        ],
    )
    html = render_html(data)
    assert "High risk" in html and "Medium risk" in html
    assert html.index("High risk") < html.index("Medium risk")
    assert 'dot red' in html      # empty high-risk field
    assert 'dot green' in html    # verified medium field


def test_render_html_shows_stp_review_banner_when_not_straight_through():
    q = QualityReport(quality_score=0.9, garble_ratio=0.0, empty_ratio=0.0,
                      table_integrity=1.0, mean_confidence=1.0, needs_review=False)
    data = ReportData(
        doc_id="d", dirty_quality=q, cleaned_quality=q, dirty_sample="", cleaned_sample="",
        fields=[
            _fr(field="total_value", value="", verified=False, reasons=["empty"], risk="high"),
            _fr(field="counterparty", value="Acme Inc.", verified=True, risk="medium"),
        ],
    )
    html = render_html(data)
    assert "Straight-through: 1/2 fields (50%)" in html
    assert "needs review: total value" in html
    assert "Straight-through document" not in html


def test_render_html_shows_straight_through_banner_when_all_green():
    q = QualityReport(quality_score=0.97, garble_ratio=0.0, empty_ratio=0.0,
                      table_integrity=1.0, mean_confidence=1.0, needs_review=False)
    data = ReportData(
        doc_id="d", dirty_quality=q, cleaned_quality=q, dirty_sample="", cleaned_sample="",
        fields=[_fr(field="counterparty", verified=True), _fr(field="governing_law", verified=True)],
    )
    html = render_html(data)
    assert "Straight-through document — no human review required." in html
    assert "stpline warn" not in html  # the review banner variant did not also render
