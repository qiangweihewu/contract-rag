"""CLM-aligned facts export: row shape, risk resolution, verify() semantics,
CSV/JSON round-trips, vertical genericity. Dep-free — hand-built facts + IR."""
from __future__ import annotations

import csv
import io
import json

import pytest
from pydantic import BaseModel, Field

from contract_rag.demo.export import (
    COLUMNS,
    clm_field_map,
    facts_rows,
    rows_from_report,
    serialize,
    to_csv,
    to_json,
)
from contract_rag.ir import BlockType, DocBlock, DocumentIR
from contract_rag.verticals.base import ExtractedClause
from contract_rag.verticals.contract.schema import ContractFacts
from contract_rag.verticals.contract.vertical import ContractVertical


def _block(text: str, bid: str) -> DocBlock:
    return DocBlock(block_id=bid, type=BlockType.PARAGRAPH, text=text,
                    confidence=1.0, source_engine="docling")


def _ir(blocks: list[DocBlock]) -> DocumentIR:
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", blocks=blocks, metadata={})


def _clause(value: str = "", bid: str | None = None, conf: float = 0.9) -> ExtractedClause:
    return ExtractedClause(value=value, source_block_id=bid, confidence=conf)


# ---------------------------------------------------------------- fake vertical

class MiniFacts(BaseModel):
    alpha: ExtractedClause = Field(default_factory=ExtractedClause)
    beta: ExtractedClause = Field(default_factory=ExtractedClause)
    gamma: ExtractedClause = Field(default_factory=ExtractedClause)


class MiniVertical:
    """Deliberately has NO field_risk seam — risk must default to medium."""

    name = "mini"
    facts_model = MiniFacts
    field_names = ("alpha", "beta", "gamma")
    set_fields = ()
    judgment_fields = ()

    def entities(self, value: str) -> list[str]:
        return []


def _mini_fixture():
    ir = _ir([_block("the alpha value lives here", "#/b/0"),
              _block("something unrelated", "#/b/1")])
    facts = MiniFacts(
        alpha=_clause("alpha value", "#/b/0", 0.9),      # attributed + confident → verified
        beta=_clause("not in any block", "#/b/1", 0.9),  # unattributed → not verified
        gamma=_clause(),                                 # empty → not verified
    )
    return facts, ir


# ---------------------------------------------------------------- row shape


def test_rows_have_all_columns_one_per_field():
    facts, ir = _mini_fixture()
    rows = facts_rows(facts, ir, "doc-1", vertical=MiniVertical())
    assert len(rows) == 3
    assert [r["field"] for r in rows] == ["alpha", "beta", "gamma"]
    for r in rows:
        assert set(r) == set(COLUMNS)
        assert r["doc_id"] == "doc-1"


def test_verified_reuses_verify_semantics():
    facts, ir = _mini_fixture()
    rows = {r["field"]: r for r in facts_rows(facts, ir, "d", vertical=MiniVertical())}
    assert rows["alpha"]["verified"] is True         # value appears in cited block
    assert rows["beta"]["verified"] is False         # unattributed
    assert rows["gamma"]["verified"] is False        # empty
    # low confidence quarantines even an attributed value (verify floor 0.6)
    low = MiniFacts(alpha=_clause("alpha value", "#/b/0", 0.3))
    row = facts_rows(low, ir, "d", vertical=MiniVertical())[0]
    assert row["verified"] is False


def test_risk_defaults_to_medium_without_field_risk_seam():
    facts, ir = _mini_fixture()
    rows = facts_rows(facts, ir, "d", vertical=MiniVertical())
    assert {r["risk_tier"] for r in rows} == {"medium"}


def test_risk_resolves_the_vertical_field_risk_seam():
    ir = _ir([_block("Total value is $5,000 payable to Acme Inc.", "#/b/0")])
    facts = ContractFacts(
        counterparty=_clause(), effective_date=_clause(), governing_law=_clause(),
        total_value=_clause("$5,000", "#/b/0", 0.9),
    )
    rows = {r["field"]: r for r in facts_rows(facts, ir, "d", vertical=ContractVertical())}
    assert rows["total_value"]["risk_tier"] == "high"
    assert rows["counterparty"]["risk_tier"] == "medium"
    assert rows["effective_date"]["risk_tier"] == "low"


def test_empty_source_block_id_serializes_as_empty_string():
    facts, ir = _mini_fixture()
    rows = {r["field"]: r for r in facts_rows(facts, ir, "d", vertical=MiniVertical())}
    assert rows["gamma"]["source_block_id"] == ""


# ---------------------------------------------------------------- CLM mapping


def test_generic_is_identity_mapping():
    facts, ir = _mini_fixture()
    rows = facts_rows(facts, ir, "d", vertical=MiniVertical(), clm="generic")
    assert all(r["clm_field"] == r["field"] for r in rows)


def test_salesforce_maps_known_fields_and_keeps_unmapped_names():
    ir = _ir([_block("x", "#/b/0")])
    facts = ContractFacts(counterparty=_clause(), effective_date=_clause(),
                          governing_law=_clause())
    rows = {r["field"]: r for r in
            facts_rows(facts, ir, "d", vertical=ContractVertical(), clm="salesforce")}
    assert rows["effective_date"]["clm_field"] == "StartDate"
    assert rows["counterparty"]["clm_field"] == "AccountName"
    assert rows["termination_notice_days"]["clm_field"] == "OwnerExpirationNotice"
    # a field absent from the map keeps our name
    mini = facts_rows(*_mini_fixture(), "d", vertical=MiniVertical(), clm="salesforce")
    assert mini[0]["clm_field"] == "alpha"


def test_ironclad_maps_to_camelcase_property_names():
    ir = _ir([_block("x", "#/b/0")])
    facts = ContractFacts(counterparty=_clause(), effective_date=_clause(),
                          governing_law=_clause())
    rows = {r["field"]: r for r in
            facts_rows(facts, ir, "d", vertical=ContractVertical(), clm="ironclad")}
    assert rows["counterparty"]["clm_field"] == "counterpartyName"
    assert rows["governing_law"]["clm_field"] == "governingLaw"


def test_unknown_clm_target_raises():
    with pytest.raises(ValueError, match="unknown CLM target"):
        clm_field_map("hubspot")
    facts, ir = _mini_fixture()
    with pytest.raises(ValueError):
        facts_rows(facts, ir, "d", vertical=MiniVertical(), clm="hubspot")


# ---------------------------------------------------------------- serializers


def test_csv_round_trip():
    facts, ir = _mini_fixture()
    rows = facts_rows(facts, ir, "doc-1", vertical=MiniVertical())
    text = to_csv(rows)
    back = list(csv.DictReader(io.StringIO(text)))
    assert len(back) == 3
    assert back[0]["field"] == "alpha"
    assert back[0]["value"] == "alpha value"
    assert back[0]["verified"] == "True" and back[1]["verified"] == "False"
    assert float(back[0]["confidence"]) == 0.9
    assert list(back[0]) == list(COLUMNS)


@pytest.mark.parametrize(
    "raw",
    ["=SUM(A1:A9)", "+1234", "-1234", "@cmd", "\tpayload", "\rpayload"],
)
def test_to_csv_escapes_formula_prefixed_values(raw):
    row = {c: "" for c in COLUMNS}
    row["value"] = raw
    text = to_csv([row])
    back = list(csv.DictReader(io.StringIO(text)))
    assert back[0]["value"] == "'" + raw       # leading quote defuses the formula
    assert back[0]["value"] != raw


def test_to_csv_leaves_benign_values_untouched():
    row = {c: "" for c in COLUMNS}
    row["value"] = "California law applies"
    row["confidence"] = 0.9
    text = to_csv([row])
    back = list(csv.DictReader(io.StringIO(text)))
    assert back[0]["value"] == "California law applies"


def test_json_round_trip_preserves_types():
    facts, ir = _mini_fixture()
    rows = facts_rows(facts, ir, "doc-1", vertical=MiniVertical())
    back = json.loads(to_json(rows))
    assert back == rows                       # bools/floats survive as-is
    assert back[0]["verified"] is True


def test_serialize_dispatches_and_rejects_unknown_format():
    facts, ir = _mini_fixture()
    rows = facts_rows(facts, ir, "d", vertical=MiniVertical())
    assert serialize(rows, "csv") == to_csv(rows)
    assert serialize(rows, "json") == to_json(rows)
    with pytest.raises(ValueError, match="unknown export format"):
        serialize(rows, "xml")


# ---------------------------------------------------------------- stp (additive)


def test_to_json_without_stp_is_byte_identical_to_before():
    facts, ir = _mini_fixture()
    rows = facts_rows(facts, ir, "d", vertical=MiniVertical())
    assert json.loads(to_json(rows)) == rows      # plain array, unchanged


def test_to_json_with_stp_wraps_facts_and_adds_top_level_stp_key():
    facts, ir = _mini_fixture()
    rows = facts_rows(facts, ir, "d", vertical=MiniVertical())
    stp = {"stp_fields": 1, "total_fields": 3, "stp_rate": 1 / 3,
           "straight_through": False, "review_fields": ["beta"]}
    back = json.loads(to_json(rows, stp=stp))
    assert back == {"facts": rows, "stp": stp}


def test_serialize_json_forwards_stp_but_csv_ignores_it():
    facts, ir = _mini_fixture()
    rows = facts_rows(facts, ir, "d", vertical=MiniVertical())
    stp = {"stp_fields": 1, "total_fields": 3, "stp_rate": 1 / 3,
           "straight_through": False, "review_fields": ["beta"]}
    assert json.loads(serialize(rows, "json", stp=stp)) == {"facts": rows, "stp": stp}
    # CSV row shape is the downstream CLM import contract — stp is a no-op for csv.
    assert serialize(rows, "csv", stp=stp) == to_csv(rows) == serialize(rows, "csv")


# ---------------------------------------------------------------- NDA vertical


def test_nda_vertical_exports_transparently():
    from contract_rag.verticals.nda.schema import NDAFacts
    from contract_rag.verticals.nda.vertical import NDAVertical

    ir = _ir([_block("This NDA is between Acme Inc. and Globex LLC.", "#/b/0"),
              _block("Governed by the laws of the State of New York.", "#/b/1")])
    facts = NDAFacts(
        disclosing_party=_clause("Acme Inc.", "#/b/0", 0.9),
        governing_law=_clause("New York", "#/b/1", 0.9),
    )
    rows = {r["field"]: r for r in facts_rows(facts, ir, "nda-1", vertical=NDAVertical())}
    assert set(rows) == set(NDAFacts.FIELD_NAMES)
    assert rows["disclosing_party"]["verified"] is True     # entity appears in cited block
    assert rows["return_of_materials"]["risk_tier"] == "high"   # NDA field_risk seam
    assert rows["effective_date"]["risk_tier"] == "low"
    # salesforce map: shared field maps, NDA-specific field keeps our name
    sf = {r["field"]: r for r in
          facts_rows(facts, ir, "nda-1", vertical=NDAVertical(), clm="salesforce")}
    assert sf["effective_date"]["clm_field"] == "StartDate"
    assert sf["disclosing_party"]["clm_field"] == "disclosing_party"


# ---------------------------------------------------------------- report adapter


def test_rows_from_report_matches_the_report_fields():
    from contract_rag.demo.report import FieldRow, ReportData
    from contract_rag.clean.quality import QualityReport

    qr = QualityReport(quality_score=0.9, garble_ratio=0.0, empty_ratio=0.0,
                       table_integrity=1.0, mean_confidence=1.0, needs_review=False)
    data = ReportData(
        doc_id="ACME_MSA",
        dirty_quality=qr, cleaned_quality=qr,
        fields=[FieldRow(field="counterparty", dirty_value="", cleaned_value="Acme Inc.",
                         source_block_id="#/b/0", confidence=0.8, verified=True,
                         reasons=[], risk="medium")],
        dirty_sample="", cleaned_sample="",
    )
    rows = rows_from_report(data, clm="ironclad")
    assert rows == [{
        "doc_id": "ACME_MSA", "field": "counterparty",
        "clm_field": "counterpartyName", "value": "Acme Inc.",
        "source_block_id": "#/b/0", "confidence": 0.8,
        "risk_tier": "medium", "verified": True,
    }]


def test_json_export_from_report_carries_the_report_stp_summary():
    """The wiring report.py's CLI uses: serialize(rows_from_report(data), "json",
    stp=stp_summary(data.fields)) — the exported stp must match what the HTML
    report's own banner/lights would show for the same fields."""
    from contract_rag.demo.report import FieldRow, ReportData, stp_summary
    from contract_rag.clean.quality import QualityReport

    qr = QualityReport(quality_score=0.9, garble_ratio=0.0, empty_ratio=0.0,
                       table_integrity=1.0, mean_confidence=1.0, needs_review=False)
    data = ReportData(
        doc_id="ACME_MSA", dirty_quality=qr, cleaned_quality=qr,
        fields=[
            FieldRow(field="counterparty", dirty_value="", cleaned_value="Acme Inc.",
                     source_block_id="#/b/0", confidence=0.8, verified=True,
                     reasons=[], risk="medium"),
            FieldRow(field="total_value", dirty_value="", cleaned_value="",
                     source_block_id=None, confidence=0.0, verified=False,
                     reasons=["empty"], risk="high"),
        ],
        dirty_sample="", cleaned_sample="",
    )
    rows = rows_from_report(data, clm="generic")
    stp = stp_summary(data.fields)
    payload = json.loads(serialize(rows, "json", stp=stp))
    assert payload["facts"] == rows
    assert payload["stp"] == {
        "stp_fields": 1, "total_fields": 2, "stp_rate": 0.5,
        "straight_through": False, "review_fields": ["total_value"],
    }


# ---------------------------------------------------------------- batch wiring


def _batch_fixture(tmp_path):
    from contract_rag.eval.golden import GoldenDoc

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    golden = []
    for i in range(2):
        (data_dir / f"doc{i}.pdf").write_bytes(b"%PDF-1.4 fake")
        golden.append(GoldenDoc(doc_id=f"doc{i}", source_pdf=f"doc{i}.pdf", facts={}))
    parse_ir = _ir([
        _block("This Agreement is by and between Acme Inc. and Globex LLC.", "#/b/0"),
        _block("Governed by the laws of the State of New York.", "#/b/1"),
    ])
    return golden, data_dir, parse_ir


def test_run_batch_export_writes_per_doc_and_combined_facts(tmp_path):
    from contract_rag.demo.batch import run_batch
    from contract_rag.extract.rules import RuleExtractor

    golden, data_dir, parse_ir = _batch_fixture(tmp_path)
    out = tmp_path / "reports"
    run_batch(golden, data_dir, out, RuleExtractor(),
              parse_fn=lambda _p: parse_ir, seed=0, export="csv", clm="salesforce")

    assert (out / "doc0.facts.csv").exists() and (out / "doc1.facts.csv").exists()
    combined = list(csv.DictReader(io.StringIO((out / "facts.csv").read_text())))
    assert {r["doc_id"] for r in combined} == {"doc0", "doc1"}
    assert len(combined) == 2 * len(ContractFacts.FIELD_NAMES)
    assert any(r["clm_field"] == "StartDate" for r in combined)   # clm mapping applied


def test_run_batch_default_writes_no_facts_files(tmp_path):
    from contract_rag.demo.batch import run_batch
    from contract_rag.extract.rules import RuleExtractor

    golden, data_dir, parse_ir = _batch_fixture(tmp_path)
    out = tmp_path / "reports"
    run_batch(golden, data_dir, out, RuleExtractor(), parse_fn=lambda _p: parse_ir, seed=0)
    assert not list(out.glob("*.facts.*")) and not (out / "facts.csv").exists()
    assert (out / "index.html").exists()                          # today's behavior intact
