from contract_rag.eval.golden import GoldenDoc, normalize
from contract_rag.eval.metrics import aggregate, field_scores, source_attribution_ok
from contract_rag.extract.schema import ContractFacts, ExtractedClause
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def test_normalize_is_lenient():
    assert normalize("  New York. ") == normalize("new york")


def test_field_scores_match_on_normalized_value():
    pred = ContractFacts(
        counterparty=ExtractedClause(value="Acme, Inc.", source_block_id="#/b/1", confidence=0.9),
        effective_date=ExtractedClause(value="2025-12-31", source_block_id="#/b/2", confidence=0.8),
        governing_law=ExtractedClause(),
    )
    gold = GoldenDoc(
        doc_id="d1", source_pdf="d1.pdf",
        facts={"counterparty": "Acme Inc", "effective_date": "2026-01-01", "governing_law": "New York"},
    )
    scores = field_scores(pred, gold)
    assert scores["counterparty"] is True
    assert scores["effective_date"] is False   # wrong value
    assert scores["governing_law"] is False     # missed (empty prediction vs non-empty gold)


def test_source_attribution_checks_value_is_in_cited_block():
    ir = DocumentIR(
        doc_id="d1", source_uri="file:///x", file_hash="h", mime_type="application/pdf",
        blocks=[
            DocBlock(block_id="#/b/1", type=BlockType.PARAGRAPH,
                     text="entered into by Acme Inc.",
                     bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                     confidence=1.0, source_engine="docling"),
        ],
        metadata={},
    )
    pred = ContractFacts(
        counterparty=ExtractedClause(value="Acme Inc.", source_block_id="#/b/1", confidence=0.9),
        effective_date=ExtractedClause(value="2026", source_block_id="#/b/1", confidence=0.7),  # not in block
        governing_law=ExtractedClause(),
    )
    ok = source_attribution_ok(pred, ir)
    assert ok["counterparty"] is True
    assert ok["effective_date"] is False


def _cp(value, gold_cp):
    pred = ContractFacts(
        counterparty=ExtractedClause(value=value, source_block_id="#/b/1", confidence=0.5),
        effective_date=ExtractedClause(), governing_law=ExtractedClause(),
    )
    gold = GoldenDoc(doc_id="d", source_pdf="d.pdf",
                     facts={"counterparty": gold_cp, "effective_date": "", "governing_law": ""})
    return field_scores(pred, gold)["counterparty"]


def test_counterparty_uses_set_overlap_not_exact_string():
    # order/extra-alias differences still count as correct via entity-set overlap
    assert _cp("Globex LLC; Acme Inc.", "Acme Inc.; Globex LLC") is True
    assert _cp("Acme Inc.", "Acme Inc.; Globex LLC") is True        # 1 of 2 == jaccard 0.5
    assert _cp("Initech Inc.", "Acme Inc.; Globex LLC") is False    # disjoint
    assert _cp("", "Acme Inc.; Globex LLC") is False                # empty pred


def test_governing_law_canonicalizes_phrasing_to_jurisdiction():
    def gl(pred_val, gold_val):
        pred = ContractFacts(
            counterparty=ExtractedClause(), effective_date=ExtractedClause(),
            governing_law=ExtractedClause(value=pred_val, source_block_id="b", confidence=0.5),
        )
        gold = GoldenDoc(doc_id="d", source_pdf="d.pdf",
                         facts={"counterparty": "", "effective_date": "", "governing_law": gold_val})
        return field_scores(pred, gold)["governing_law"]

    # the metric is backend-agnostic: any phrasing reduces to the jurisdiction
    assert gl("State of New York", "New York") is True
    assert gl("the laws of the State of Delaware", "Delaware") is True
    assert gl("New York", "New York") is True            # already-canonical (rule backend)
    assert gl("California", "Texas") is False
    assert gl("", "New York") is False


def test_source_attribution_counterparty_requires_all_entities_in_block():
    def ir_with(text):
        return DocumentIR(
            doc_id="d", source_uri="file:///x", file_hash="h", mime_type="application/pdf",
            blocks=[DocBlock(block_id="#/b/1", type=BlockType.PARAGRAPH, text=text,
                             bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                             confidence=1.0, source_engine="docling")],
            metadata={})

    pred = ContractFacts(
        counterparty=ExtractedClause(value="Acme Inc.; Globex LLC", source_block_id="#/b/1", confidence=0.5),
        effective_date=ExtractedClause(), governing_law=ExtractedClause(),
    )
    assert source_attribution_ok(pred, ir_with("between Acme Inc. and Globex LLC"))["counterparty"] is True
    assert source_attribution_ok(pred, ir_with("between Acme Inc. only"))["counterparty"] is False


def _empties(*names):
    return {n: False for n in names}


def test_termination_notice_days_matches_numerically():
    def t(pred_val, gold_val):
        pred = ContractFacts(
            counterparty=ExtractedClause(), effective_date=ExtractedClause(), governing_law=ExtractedClause(),
            termination_notice_days=ExtractedClause(value=pred_val, source_block_id="b", confidence=0.9),
        )
        gold = GoldenDoc(doc_id="d", source_pdf="d.pdf", facts={"termination_notice_days": gold_val})
        return field_scores(pred, gold)["termination_notice_days"]

    assert t("ninety (90) days", "90") is True
    assert t("90 days prior written notice", "90") is True
    assert t("30 days", "90") is False
    assert t("", "90") is False


def test_auto_renewal_matches_yes_no_and_is_attribution_exempt():
    def mk(val):
        return ContractFacts(
            counterparty=ExtractedClause(), effective_date=ExtractedClause(), governing_law=ExtractedClause(),
            auto_renewal=ExtractedClause(value=val, source_block_id="b", confidence=0.9),
        )

    def g(gv):
        return GoldenDoc(doc_id="d", source_pdf="d.pdf", facts={"auto_renewal": gv})

    assert field_scores(mk("yes"), g("yes"))["auto_renewal"] is True
    assert field_scores(mk("Yes, it renews automatically"), g("yes"))["auto_renewal"] is True
    assert field_scores(mk("no"), g("yes"))["auto_renewal"] is False

    # judgment field: "yes" need not literally appear in the cited block
    ir = DocumentIR(
        doc_id="d", source_uri="x", file_hash="h", mime_type="application/pdf",
        blocks=[DocBlock(block_id="b", type=BlockType.PARAGRAPH, text="renews automatically",
                         bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                         confidence=1.0, source_engine="x")],
        metadata={})
    assert source_attribution_ok(mk("yes"), ir)["auto_renewal"] is True


def test_aggregate_excludes_fields_with_no_gold():
    F = ContractFacts.FIELD_NAMES

    def row(scores, pred, gold):
        return {"scores": {n: scores.get(n, False) for n in F},
                "pred_nonempty": {n: pred.get(n, False) for n in F},
                "gold_nonempty": {n: gold.get(n, False) for n in F},
                "source": {n: scores.get(n, False) for n in F}}

    # counterparty correct; total_value predicted but has no gold anywhere
    rows = [row({"counterparty": True},
                {"counterparty": True, "total_value": True},
                {"counterparty": True})]
    agg = aggregate(rows)
    assert "total_value" not in agg["scored_fields"]
    assert agg["support"]["total_value"] == 0
    assert agg["precision"] == 1.0   # the unjudgeable total_value prediction is not a FP
    assert agg["per_field_on_labeled"]["total_value"] is None
    assert agg["per_field_on_labeled"]["counterparty"] == 1.0


def test_aggregate_computes_f1_and_source_accuracy():
    yes = {n: True for n in ContractFacts.FIELD_NAMES}
    rows = [{"scores": dict(yes), "source": dict(yes), "pred_nonempty": dict(yes), "gold_nonempty": dict(yes)}]
    rows[0]["scores"]["effective_date"] = False
    agg = aggregate(rows)
    assert 0.0 <= agg["field_f1"] <= 1.0
    assert agg["per_field"]["counterparty"] == 1.0
    assert 0.0 <= agg["source_accuracy"] <= 1.0
