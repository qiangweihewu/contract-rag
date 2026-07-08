from __future__ import annotations

from contract_rag.verticals.contract.gold import normalize_facts


def test_normalize_facts_reduces_spans_to_answer_space():
    raw = {
        "governing_law": "['This Agreement is governed by the laws of the State of New York.']",
        "termination_notice_days": "['upon ninety (90) days written notice']",
    }
    out = normalize_facts(raw)
    assert out["governing_law"] == "New York"
    assert out["termination_notice_days"] == "90"
