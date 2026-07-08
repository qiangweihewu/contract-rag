from __future__ import annotations

import pytest
from pydantic import ValidationError

from contract_rag.extract.schema import ContractFacts, ExtractedClause
from contract_rag.verticals.base import ExtractedClause as BaseClause
from contract_rag.verticals.base import Vertical


def test_extracted_clause_defaults():
    c = BaseClause()
    assert c.value == "" and c.source_block_id is None and c.confidence == 0.0


def test_schema_reexports_the_same_class():
    assert ExtractedClause is BaseClause


def test_contract_facts_constructs_empty():
    # bare ContractFacts() must raise — 3 required fields (counterparty, effective_date,
    # governing_law) have no default; use ContractVertical().empty_facts() instead.
    with pytest.raises(ValidationError):
        ContractFacts()
    from contract_rag.verticals.contract.vertical import ContractVertical
    ef = ContractVertical().empty_facts()
    assert ef.counterparty.value == "" and ef.governing_law.value == ""


def test_vertical_protocol_is_runtime_checkable():
    class Dummy:
        name = "x"; facts_model = ContractFacts
        field_names = (); set_fields = (); judgment_fields = ()
        extraction_prompt = ""; rule_extractor = object()
        def classify_clause(self, chunk): return "other"
        def permission_tags(self, chunk): return []
        def normalize_gold(self, raw): return dict(raw)
        def canonicalize_value(self, name, value): return value
        def entities(self, value): return []
        def empty_facts(self): return self.facts_model()  # body never called; @runtime_checkable only checks attribute presence
    assert isinstance(Dummy(), Vertical)


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        BaseClause(confidence=5.0)
    with pytest.raises(ValidationError):
        BaseClause(confidence=-0.1)
