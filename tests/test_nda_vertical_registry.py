from __future__ import annotations

from contract_rag.verticals.base import Vertical
from contract_rag.verticals.nda.schema import NDAFacts
from contract_rag.verticals.nda.vertical import NDAVertical
from contract_rag.verticals.registry import get_vertical


def test_nda_vertical_satisfies_protocol():
    v = NDAVertical()
    assert isinstance(v, Vertical)
    assert v.name == "nda"
    assert v.facts_model is NDAFacts
    assert v.field_names[0] == "disclosing_party"


def test_registered_as_builtin():
    assert get_vertical("nda").name == "nda"


def test_canon_entities_empty():
    v = NDAVertical()
    assert v.canonicalize_value("term", "two (2) years") == "2 years"
    assert v.canonicalize_value("governing_law", "the State of New York") == "New York"
    assert v.entities("Acme Robotics Inc. and Beilan Systems LLC")
    assert v.empty_facts().disclosing_party.value == ""


def test_contract_still_registered():
    # zero-core-fork sanity: adding nda did not disturb the contract vertical
    assert get_vertical("contract").name == "contract"
