from __future__ import annotations

from contract_rag.verticals.base import ExtractedClause
from contract_rag.verticals.nda.schema import NDAFacts


def test_field_metadata():
    assert NDAFacts.FIELD_NAMES == (
        "disclosing_party", "receiving_party", "effective_date", "term",
        "confidentiality_period", "return_of_materials", "governing_law",
    )
    assert NDAFacts.SET_FIELDS == ("disclosing_party", "receiving_party")
    assert NDAFacts.JUDGMENT_FIELDS == ("return_of_materials",)


def test_constructs_empty_and_from_clauses():
    assert NDAFacts().disclosing_party.value == ""
    f = NDAFacts(governing_law=ExtractedClause(value="New York", source_block_id="#/b/1", confidence=0.7))
    assert f.governing_law.value == "New York"
