from contract_rag.extract.schema import ContractFacts, ExtractedClause
from contract_rag.extract.verify import CONFIDENCE_THRESHOLD, resolve_thresholds, verify
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _b(text, bid):
    return DocBlock(block_id=bid, type=BlockType.PARAGRAPH, text=text,
                    bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                    confidence=1.0, source_engine="x")


def _ir(blocks):
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", blocks=blocks, metadata={})


def _facts(cp=None, ed=None, gl=None):
    e = ExtractedClause
    return ContractFacts(counterparty=cp or e(), effective_date=ed or e(), governing_law=gl or e())


def test_verify_passes_attributed_high_confidence_value():
    ir = _ir([_b("Governed by the laws of the State of New York.", "#/b/1")])
    facts = _facts(gl=ExtractedClause(value="New York", source_block_id="#/b/1", confidence=0.9))
    rep = verify(facts, ir)
    assert rep.checks["governing_law"].passed is True
    assert "governing_law" in rep.verified
    assert "governing_law" not in rep.quarantined


def test_verify_quarantines_unattributed_value():
    ir = _ir([_b("Governed by the laws of the State of New York.", "#/b/1")])
    facts = _facts(gl=ExtractedClause(value="California", source_block_id="#/b/1", confidence=0.9))
    c = verify(facts, ir).checks["governing_law"]
    assert c.passed is False
    assert c.attributed is False
    assert "unattributed" in c.reasons


def test_verify_quarantines_low_confidence_value():
    ir = _ir([_b("Governed by the laws of the State of New York.", "#/b/1")])
    facts = _facts(gl=ExtractedClause(value="New York", source_block_id="#/b/1", confidence=0.4))
    c = verify(facts, ir).checks["governing_law"]
    assert c.passed is False
    assert c.attributed is True             # it IS in the block...
    assert c.reasons == ["low_confidence"]  # ...just under threshold


def test_verify_empty_field_is_neither_verified_nor_quarantined():
    rep = verify(_facts(), _ir([_b("nothing here", "#/b/1")]))
    assert rep.checks["counterparty"].reasons == ["empty"]
    assert "counterparty" not in rep.verified
    assert "counterparty" not in rep.quarantined


# ---------------------------------------------------- per-risk-tier threshold seam
# (contract vertical field_risk: total_value=high, governing_law=medium,
#  effective_date=low — see verticals/contract/vertical.py)


def test_resolve_thresholds_default_is_flat_constant():
    from contract_rag.verticals.registry import default_vertical

    v = default_vertical()
    floors = resolve_thresholds(v, CONFIDENCE_THRESHOLD, None)
    assert set(floors) == set(v.field_names)
    assert all(t == CONFIDENCE_THRESHOLD for t in floors.values())


def test_resolve_thresholds_maps_tiers_via_field_risk():
    from contract_rag.verticals.registry import default_vertical

    floors = resolve_thresholds(
        default_vertical(), 0.6, {"high": 0.9, "medium": 0.7, "low": 0.5}
    )
    assert floors["total_value"] == 0.9
    assert floors["governing_law"] == 0.7
    assert floors["effective_date"] == 0.5


def test_resolve_thresholds_missing_tier_falls_back_to_default():
    from contract_rag.verticals.registry import default_vertical

    floors = resolve_thresholds(default_vertical(), 0.6, {"high": 0.9})
    assert floors["total_value"] == 0.9
    assert floors["governing_law"] == 0.6  # medium not in map -> flat default


def test_resolve_thresholds_vertical_without_field_risk_is_all_medium():
    class NoRisk:
        field_names = ("a", "b")

    floors = resolve_thresholds(NoRisk(), 0.6, {"high": 0.9, "medium": 0.75})
    assert floors == {"a": 0.75, "b": 0.75}


def test_verify_without_tier_thresholds_is_unchanged():
    ir = _ir([_b("Governed by the laws of the State of New York.", "#/b/1")])
    facts = _facts(gl=ExtractedClause(value="New York", source_block_id="#/b/1", confidence=0.65))
    assert verify(facts, ir).checks["governing_law"].passed is True  # 0.65 >= 0.6 flat


def test_verify_tier_thresholds_quarantine_medium_risk_field_below_tier_floor():
    ir = _ir([_b("Governed by the laws of the State of New York.", "#/b/1")])
    facts = _facts(gl=ExtractedClause(value="New York", source_block_id="#/b/1", confidence=0.65))
    c = verify(facts, ir, tier_thresholds={"medium": 0.8}).checks["governing_law"]
    assert c.passed is False
    assert c.reasons == ["low_confidence"]  # 0.65 < the medium-tier floor 0.8


def test_verify_tier_thresholds_can_lower_a_low_risk_floor():
    ir = _ir([_b("effective as of January 1, 2024", "#/b/1")])
    facts = _facts(ed=ExtractedClause(value="January 1, 2024", source_block_id="#/b/1", confidence=0.5))
    assert verify(facts, ir).checks["effective_date"].passed is False  # flat 0.6
    rep = verify(facts, ir, tier_thresholds={"low": 0.4})
    assert rep.checks["effective_date"].passed is True


def test_verify_counterparty_requires_every_entity_in_cited_block():
    ir = _ir([_b("by and between Acme Inc. and Globex LLC", "#/b/1")])
    ok = _facts(cp=ExtractedClause(value="Acme Inc.; Globex LLC", source_block_id="#/b/1", confidence=0.9))
    assert verify(ok, ir).checks["counterparty"].passed is True
    bad = _facts(cp=ExtractedClause(value="Acme Inc.; Initech LLC", source_block_id="#/b/1", confidence=0.9))
    c = verify(bad, ir).checks["counterparty"]
    assert c.passed is False
    assert "unattributed" in c.reasons      # Initech LLC is not in the block
