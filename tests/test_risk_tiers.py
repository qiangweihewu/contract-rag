"""Field risk tiers: verticals declare which fields are high/medium/low risk
(an optional seam — resolved defensively so third-party verticals without it keep
working) and aggregate() rolls the taxonomy + F1 up per tier, so reports can say
"high-risk fields: X% omission, Y% invention" instead of one blended number."""
from __future__ import annotations

from contract_rag.eval.metrics import aggregate, field_risk_map
from contract_rag.verticals.contract.schema import ContractFacts
from contract_rag.verticals.registry import get_vertical
from tests.verticals.memo import MemoVertical

F = ContractFacts.FIELD_NAMES


# ---------------------------------------------------------- field_risk_map

def test_contract_risk_assignments():
    risk = field_risk_map(get_vertical("contract"))
    assert risk == {
        "total_value": "high", "termination_notice_days": "high", "auto_renewal": "high",
        "counterparty": "medium", "governing_law": "medium",
        "effective_date": "low",
    }


def test_nda_risk_assignments():
    risk = field_risk_map(get_vertical("nda"))
    assert risk == {
        "return_of_materials": "high", "confidentiality_period": "high", "term": "high",
        "disclosing_party": "medium", "receiving_party": "medium", "governing_law": "medium",
        "effective_date": "low",
    }


def test_vertical_without_field_risk_defaults_all_medium():
    # MemoVertical predates the seam — the resolver must not require it
    risk = field_risk_map(MemoVertical())
    assert risk == {"author": "medium", "date": "medium"}


def test_partial_or_invalid_mapping_falls_back_to_medium():
    class Partial(MemoVertical):
        def field_risk(self):
            return {"author": "high", "date": "critical"}  # 'critical' is not a level

    risk = field_risk_map(Partial())
    assert risk == {"author": "high", "date": "medium"}

    class AttrStyle(MemoVertical):
        field_risk = {"date": "low"}  # plain mapping, not a method

    assert field_risk_map(AttrStyle()) == {"author": "medium", "date": "low"}


# ---------------------------------------------------------- aggregate rollup

def _row(scores=None, pred=None, gold=None):
    scores, pred, gold = scores or {}, pred or {}, gold or {}
    return {
        "scores": {n: scores.get(n, False) for n in F},
        "pred_nonempty": {n: pred.get(n, False) for n in F},
        "gold_nonempty": {n: gold.get(n, False) for n in F},
        "source": {n: scores.get(n, False) for n in F},
    }


def test_aggregate_rolls_up_per_tier():
    rows = [
        # high tier: auto_renewal invented, termination omitted; medium: counterparty correct
        _row(scores={"counterparty": True},
             pred={"counterparty": True, "auto_renewal": True},
             gold={"counterparty": True, "auto_renewal": True,
                   "termination_notice_days": True}),
    ]
    agg = aggregate(rows)
    tiers = agg["risk_tiers"]
    assert tiers["field_risk"]["auto_renewal"] == "high"

    high = tiers["per_tier"]["high"]
    assert set(high["fields"]) == {"total_value", "termination_notice_days", "auto_renewal"}
    assert high["support"] == 2
    assert high["taxonomy"]["invention"] == 1
    assert high["taxonomy"]["omission"] == 1
    assert high["taxonomy"]["correct"] == 0
    assert high["f1_on_labeled"] == 0.0

    med = tiers["per_tier"]["medium"]
    assert med["taxonomy"]["correct"] == 1
    assert med["f1_on_labeled"] == 1.0

    # low tier (effective_date) has no gold in this set → unmeasurable, not 0
    assert tiers["per_tier"]["low"]["f1_on_labeled"] is None
    assert tiers["per_tier"]["low"]["support"] == 0


def test_per_tier_f1_matches_global_when_single_tier_carries_all_gold():
    rows = [_row(scores={"governing_law": True}, pred={"governing_law": True},
                 gold={"governing_law": True})]
    agg = aggregate(rows)
    assert agg["risk_tiers"]["per_tier"]["medium"]["f1_on_labeled"] == agg["field_f1"]


def test_aggregate_tiers_with_injected_riskless_vertical():
    memo = MemoVertical()
    rows = [{
        "scores": {"author": True, "date": False},
        "pred_nonempty": {"author": True, "date": False},
        "gold_nonempty": {"author": True, "date": True},
        "source": {"author": True, "date": False},
    }]
    agg = aggregate(rows, vertical=memo)
    per_tier = agg["risk_tiers"]["per_tier"]
    assert set(per_tier) == {"medium"}          # everything defaults to medium
    assert per_tier["medium"]["taxonomy"] == {
        "correct": 1, "omission": 1, "invention": 0, "unscored": 0, "wrong_span": 0}
