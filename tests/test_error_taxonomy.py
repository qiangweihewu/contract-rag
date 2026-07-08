"""Error taxonomy: every (doc, field) with labeled gold is exactly one of
correct / omission / invention; unlabeled fields with a prediction are `unscored`
(never invention — the zero-gold honesty rule); `wrong_span` is an orthogonal flag
for correct values whose source-attribution fails."""
from __future__ import annotations

from contract_rag.eval.golden import GoldenDoc
from contract_rag.eval.metrics import aggregate, classify_field, row_for
from contract_rag.ir import BlockType, DocBlock, DocumentIR
from contract_rag.verticals.base import ExtractedClause
from contract_rag.verticals.contract.schema import ContractFacts
from contract_rag.verticals.registry import get_vertical

F = ContractFacts.FIELD_NAMES


# ---------------------------------------------------------- classify_field

def test_classify_correct():
    assert classify_field(gold_nonempty=True, pred_nonempty=True, correct=True) == "correct"


def test_classify_omission_gold_labeled_prediction_empty():
    assert classify_field(gold_nonempty=True, pred_nonempty=False, correct=False) == "omission"


def test_classify_invention_gold_labeled_prediction_wrong():
    assert classify_field(gold_nonempty=True, pred_nonempty=True, correct=False) == "invention"


def test_classify_unscored_no_gold_but_predicted():
    # unverifiable, NOT an invention — keeps the zero-gold honesty rule
    assert classify_field(gold_nonempty=False, pred_nonempty=True, correct=False) == "unscored"


def test_classify_nothing_when_no_gold_and_no_prediction():
    assert classify_field(gold_nonempty=False, pred_nonempty=False, correct=False) is None


# ---------------------------------------------------------- aggregate rollup

def _row(scores=None, pred=None, gold=None, source=None):
    scores, pred, gold, source = scores or {}, pred or {}, gold or {}, source or {}
    return {
        "scores": {n: scores.get(n, False) for n in F},
        "pred_nonempty": {n: pred.get(n, False) for n in F},
        "gold_nonempty": {n: gold.get(n, False) for n in F},
        # default: attribution mirrors correctness (the common case)
        "source": {n: source.get(n, scores.get(n, False)) for n in F},
    }


def test_aggregate_error_taxonomy_counts_per_field_and_totals():
    rows = [
        # doc 1: counterparty correct; governing_law invented; effective_date omitted
        _row(scores={"counterparty": True},
             pred={"counterparty": True, "governing_law": True},
             gold={"counterparty": True, "governing_law": True, "effective_date": True}),
        # doc 2: total_value predicted but never gold-labeled → unscored
        _row(pred={"total_value": True}, gold={"counterparty": True}),
    ]
    tax = aggregate(rows)["error_taxonomy"]
    assert tax["per_field"]["counterparty"] == {
        "correct": 1, "omission": 1, "invention": 0, "unscored": 0, "wrong_span": 0}
    assert tax["per_field"]["governing_law"]["invention"] == 1
    assert tax["per_field"]["effective_date"]["omission"] == 1
    assert tax["per_field"]["total_value"] == {
        "correct": 0, "omission": 0, "invention": 0, "unscored": 1, "wrong_span": 0}
    assert tax["totals"] == {
        "correct": 1, "omission": 2, "invention": 1, "unscored": 1, "wrong_span": 0}


def test_aggregate_wrong_span_is_orthogonal_to_correct():
    # correct value, but the cited block doesn't contain it → correct AND wrong_span
    rows = [_row(scores={"governing_law": True},
                 pred={"governing_law": True},
                 gold={"governing_law": True},
                 source={"governing_law": False})]
    tax = aggregate(rows)["error_taxonomy"]
    assert tax["per_field"]["governing_law"]["correct"] == 1
    assert tax["per_field"]["governing_law"]["wrong_span"] == 1
    assert tax["totals"]["wrong_span"] == 1


def test_aggregate_existing_keys_unchanged():
    rows = [_row(scores={"counterparty": True}, pred={"counterparty": True},
                 gold={"counterparty": True})]
    agg = aggregate(rows)
    for key in ("field_f1", "precision", "recall", "per_field", "per_field_on_labeled",
                "support", "scored_fields", "source_accuracy", "n_docs"):
        assert key in agg
    assert agg["precision"] == 1.0 and agg["n_docs"] == 1


# ------------------------------------------------- end-to-end via row_for (set field)

def _ir(text: str) -> DocumentIR:
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", metadata={}, blocks=[
        DocBlock(block_id="b1", type=BlockType.PARAGRAPH, text=text,
                 confidence=1.0, source_engine="docling")])


def test_set_field_overlap_pass_counts_as_correct():
    v = get_vertical("contract")
    pred = ContractFacts(
        counterparty=ExtractedClause(value="Acme Inc.", source_block_id="b1", confidence=0.9),
        effective_date=ExtractedClause(), governing_law=ExtractedClause(),
    )
    gold = GoldenDoc(doc_id="d", source_pdf="d.pdf",
                     facts={"counterparty": "Acme Inc.; Globex LLC"})  # jaccard 0.5 → pass
    tax = aggregate([row_for(pred, gold, _ir("between Acme Inc. and Globex LLC"), v)],
                    vertical=v)["error_taxonomy"]
    assert tax["per_field"]["counterparty"]["correct"] == 1
    assert tax["per_field"]["counterparty"]["wrong_span"] == 0


def test_set_field_disjoint_counts_as_invention():
    v = get_vertical("contract")
    pred = ContractFacts(
        counterparty=ExtractedClause(value="Initech Inc.", source_block_id="b1", confidence=0.9),
        effective_date=ExtractedClause(), governing_law=ExtractedClause(),
    )
    gold = GoldenDoc(doc_id="d", source_pdf="d.pdf",
                     facts={"counterparty": "Acme Inc.; Globex LLC"})
    tax = aggregate([row_for(pred, gold, _ir("between Initech Inc. and others"), v)],
                    vertical=v)["error_taxonomy"]
    assert tax["per_field"]["counterparty"]["invention"] == 1
