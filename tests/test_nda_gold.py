from __future__ import annotations

from contract_rag.verticals.nda.gold import normalize_facts


def test_normalize_reduces_to_answer_space():
    out = normalize_facts({
        "disclosing_party": "Acme Robotics Inc.",
        "term": "two (2) years",
        "governing_law": "the laws of the State of New York",
        "return_of_materials": "yes",
        "effective_date": "March 3, 2025",
    })
    assert out["disclosing_party"] == "Acme Robotics Inc"
    assert out["term"] == "2 years"
    assert out["governing_law"] == "New York"
    assert out["return_of_materials"] == "yes"
    assert out["effective_date"] == "March 3, 2025"
