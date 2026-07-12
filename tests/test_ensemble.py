"""Unit tests for the field-level ensemble extraction backend
(`contract_rag.extract.ensemble`). Fake children are injected through the
`children=` DI seam so these tests never build a real rule/constrained backend
or touch a network. Covers: per-field routing (default table), fallback to the
other child when the routed one is empty (and fallback counting), the
`ENSEMBLE_ROUTING` env-var override (both the pure parser and end-to-end via
`Settings`), routing-map genericity over a second vertical (NDA — all fields
default to `rule` since none are in `DEFAULT_ROUTING`), and the default-on
re-attribution post-pass on the merged result."""
from __future__ import annotations

import importlib.util

import pytest

from contract_rag.config import Settings
from contract_rag.extract.ensemble import (
    DEFAULT_ROUTING,
    EnsembleExtractor,
    parse_routing_env,
    resolve_routing,
)
from contract_rag.extract.schema import ContractFacts, ExtractedClause
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR
from contract_rag.verticals.nda.schema import NDAFacts
from contract_rag.verticals.registry import get_vertical


def _block(block_id: str, text: str) -> DocBlock:
    return DocBlock(
        block_id=block_id, type=BlockType.PARAGRAPH, text=text,
        bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
        confidence=1.0, source_engine="docling",
    )


def _ir(*blocks: DocBlock) -> DocumentIR:
    return DocumentIR(
        doc_id="d1", source_uri="file:///x.pdf", file_hash="h",
        mime_type="application/pdf", blocks=list(blocks) or [_block("b0", "text")],
        metadata={},
    )


def _cf(**clauses) -> ContractFacts:
    base = {n: ExtractedClause() for n in ContractFacts.FIELD_NAMES}
    base.update(clauses)
    return ContractFacts(**base)


def _ndaf(**clauses) -> NDAFacts:
    base = {n: ExtractedClause() for n in NDAFacts.FIELD_NAMES}
    base.update(clauses)
    return NDAFacts(**base)


class _Fake:
    def __init__(self, facts):
        self._facts = facts

    def extract(self, ir):
        return self._facts


CONTRACT = get_vertical("contract")


# --- routing -------------------------------------------------------------


def test_default_routing_matches_the_measured_table():
    routing = resolve_routing(CONTRACT)
    assert routing == {
        "counterparty": "constrained", "effective_date": "rule", "governing_law": "rule",
        "total_value": "constrained", "termination_notice_days": "rule", "auto_renewal": "constrained",
    }
    assert routing == {n: DEFAULT_ROUTING.get(n, "rule") for n in CONTRACT.field_names}


def test_routes_each_field_to_its_configured_child():
    rule_facts = _cf(
        effective_date=ExtractedClause(value="2020-01-01", source_block_id="b0", confidence=0.9),
        governing_law=ExtractedClause(value="New York", source_block_id="b0", confidence=0.9),
        termination_notice_days=ExtractedClause(value="30", source_block_id="b0", confidence=0.9),
    )
    constrained_facts = _cf(
        counterparty=ExtractedClause(value="Acme Inc.", source_block_id="b0", confidence=0.9),
        total_value=ExtractedClause(value="$1,000", source_block_id="b0", confidence=0.9),
        auto_renewal=ExtractedClause(value="yes", source_block_id="b0", confidence=0.9),
    )
    ens = EnsembleExtractor(
        Settings(extract_backend="ensemble"), vertical=CONTRACT,
        children={"rule": _Fake(rule_facts), "constrained": _Fake(constrained_facts)},
        reattribute=False,
    )

    out = ens.extract(_ir(_block("b0", "irrelevant")))

    assert out.effective_date.value == "2020-01-01"
    assert out.governing_law.value == "New York"
    assert out.termination_notice_days.value == "30"
    assert out.counterparty.value == "Acme Inc."
    assert out.total_value.value == "$1,000"
    assert out.auto_renewal.value == "yes"
    assert ens.last_fallbacks == {}


# --- fallback --------------------------------------------------------------


def test_falls_back_to_the_other_child_when_the_routed_one_is_empty():
    # effective_date routes to "rule" by default; rule comes back empty here.
    rule_facts = _cf(effective_date=ExtractedClause())
    constrained_facts = _cf(
        effective_date=ExtractedClause(value="2021-05-05", source_block_id="b0", confidence=0.8),
    )
    ens = EnsembleExtractor(
        Settings(extract_backend="ensemble"), vertical=CONTRACT,
        children={"rule": _Fake(rule_facts), "constrained": _Fake(constrained_facts)},
        reattribute=False,
    )

    out = ens.extract(_ir(_block("b0", "irrelevant")))

    assert out.effective_date.value == "2021-05-05"
    assert ens.last_fallbacks == {"effective_date": 1}


def test_no_fallback_counted_when_both_children_are_empty():
    ens = EnsembleExtractor(
        Settings(extract_backend="ensemble"), vertical=CONTRACT,
        children={"rule": _Fake(_cf()), "constrained": _Fake(_cf())},
        reattribute=False,
    )

    out = ens.extract(_ir(_block("b0", "irrelevant")))

    assert out.effective_date.value == ""
    assert ens.last_fallbacks == {}


def test_no_fallback_when_routed_child_already_populated():
    rule_facts = _cf(
        effective_date=ExtractedClause(value="2020-01-01", source_block_id="b0", confidence=0.9),
    )
    constrained_facts = _cf(
        effective_date=ExtractedClause(value="9999-99-99", source_block_id="b0", confidence=0.1),
    )
    ens = EnsembleExtractor(
        Settings(extract_backend="ensemble"), vertical=CONTRACT,
        children={"rule": _Fake(rule_facts), "constrained": _Fake(constrained_facts)},
        reattribute=False,
    )

    out = ens.extract(_ir(_block("b0", "irrelevant")))

    assert out.effective_date.value == "2020-01-01"   # routed child's own value wins
    assert ens.last_fallbacks == {}


# --- ENSEMBLE_ROUTING override ---------------------------------------------


def test_parse_routing_env_parses_field_equals_backend_pairs():
    assert parse_routing_env("counterparty=constrained,effective_date=rule") == {
        "counterparty": "constrained", "effective_date": "rule",
    }


def test_parse_routing_env_handles_blank_and_malformed_input():
    assert parse_routing_env(None) == {}
    assert parse_routing_env("") == {}
    assert parse_routing_env("noequalssign,counterparty=constrained,=rule,x=") == {
        "counterparty": "constrained",
    }


def test_resolve_routing_override_ignores_unknown_fields():
    routing = resolve_routing(CONTRACT, {"counterparty": "rule", "not_a_real_field": "constrained"})
    assert routing["counterparty"] == "rule"
    assert "not_a_real_field" not in routing


def test_ensemble_routing_env_var_flows_through_settings():
    rule_facts = _cf(
        counterparty=ExtractedClause(value="Rule Corp.", source_block_id="b0", confidence=0.9),
    )
    constrained_facts = _cf(
        counterparty=ExtractedClause(value="Constrained Corp.", source_block_id="b0", confidence=0.9),
    )
    settings = Settings(extract_backend="ensemble", ensemble_routing="counterparty=rule")
    ens = EnsembleExtractor(
        settings, vertical=CONTRACT,
        children={"rule": _Fake(rule_facts), "constrained": _Fake(constrained_facts)},
        reattribute=False,
    )

    out = ens.extract(_ir(_block("b0", "irrelevant")))

    assert out.counterparty.value == "Rule Corp."   # overridden away from the default "constrained"


def test_explicit_routing_kwarg_wins_over_settings_env():
    settings = Settings(extract_backend="ensemble", ensemble_routing="counterparty=rule")
    rule_facts = _cf(counterparty=ExtractedClause(value="Rule Corp.", source_block_id="b0", confidence=0.9))
    constrained_facts = _cf(
        counterparty=ExtractedClause(value="Constrained Corp.", source_block_id="b0", confidence=0.9)
    )
    ens = EnsembleExtractor(
        settings, vertical=CONTRACT,
        children={"rule": _Fake(rule_facts), "constrained": _Fake(constrained_facts)},
        routing={"counterparty": "constrained"},
        reattribute=False,
    )

    out = ens.extract(_ir(_block("b0", "irrelevant")))

    assert out.counterparty.value == "Constrained Corp."


# --- vertical genericity -----------------------------------------------------


def test_unknown_fields_default_to_rule_child_for_a_new_vertical():
    nda = get_vertical("nda")
    routing = resolve_routing(nda)
    assert set(routing.values()) == {"rule"}   # none of NDA's fields are in DEFAULT_ROUTING

    rule_facts = _ndaf(
        disclosing_party=ExtractedClause(value="Acme Inc.", source_block_id="b0", confidence=0.9),
    )
    constrained_facts = _ndaf(
        disclosing_party=ExtractedClause(value="Should Not Win", source_block_id="b0", confidence=0.9),
    )
    ens = EnsembleExtractor(
        Settings(extract_backend="ensemble"), vertical=nda,
        children={"rule": _Fake(rule_facts), "constrained": _Fake(constrained_facts)},
        reattribute=False,
    )

    out = ens.extract(_ir(_block("b0", "irrelevant")))

    assert isinstance(out, NDAFacts)
    assert out.disclosing_party.value == "Acme Inc."


# --- re-attribution wiring ----------------------------------------------------


def test_reattribution_runs_by_default_on_the_merged_result():
    ir = _ir(
        _block("b0", "Recitals."),
        _block("b1", "The parties are Acme Inc."),
    )
    # constrained wins counterparty by default routing, but cites the wrong block.
    constrained_facts = _cf(
        counterparty=ExtractedClause(value="Acme Inc.", source_block_id="b0", confidence=0.9),
    )
    ens = EnsembleExtractor(
        Settings(extract_backend="ensemble"), vertical=CONTRACT,
        children={"rule": _Fake(_cf()), "constrained": _Fake(constrained_facts)},
    )  # reattribute defaults True

    out = ens.extract(ir)

    assert out.counterparty.source_block_id == "b1"
    assert ens.last_reattributions == {"counterparty": 1}


def test_reattribution_disabled_leaves_wrong_span_uncorrected():
    ir = _ir(
        _block("b0", "Recitals."),
        _block("b1", "The parties are Acme Inc."),
    )
    constrained_facts = _cf(
        counterparty=ExtractedClause(value="Acme Inc.", source_block_id="b0", confidence=0.9),
    )
    ens = EnsembleExtractor(
        Settings(extract_backend="ensemble"), vertical=CONTRACT,
        children={"rule": _Fake(_cf()), "constrained": _Fake(constrained_facts)},
        reattribute=False,
    )

    out = ens.extract(ir)

    assert out.counterparty.source_block_id == "b0"
    assert ens.last_reattributions == {}


# --- routing wiring in get_extractor() ---------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("openai") is None,
    reason="get_extractor builds a real ConstrainedExtractor child (needs openai installed)",
)
def test_get_extractor_routes_ensemble():
    from contract_rag.extract.extractor import get_extractor

    ex = get_extractor(Settings(extract_backend="ensemble"))
    assert isinstance(ex, EnsembleExtractor)
