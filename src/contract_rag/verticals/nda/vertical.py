from __future__ import annotations

from collections.abc import Mapping

from contract_rag.chunk.models import Chunk
from contract_rag.verticals.legal_common import jurisdiction_in, party_entities
from contract_rag.verticals.nda import enrich as _enrich
from contract_rag.verticals.nda import gold as _gold
from contract_rag.verticals.nda import rules as _rules
from contract_rag.verticals.nda.prompt import EXTRACTION_PROMPT
from contract_rag.verticals.nda.schema import NDAFacts


def _yesno(v: str) -> str:
    t = v.strip().lower()
    if not t:
        return ""
    if "yes" in t or "true" in t or "return" in t or "destroy" in t:
        return "yes"
    if "no" in t or "false" in t:
        return "no"
    return "no"


_SCALAR_CANON = {
    "effective_date": lambda v: v,
    "term": _rules.duration_in,
    "confidentiality_period": _rules.duration_in,
    "return_of_materials": _yesno,
    "governing_law": lambda v: jurisdiction_in(v) or "",
}


class NDAVertical:
    name = "nda"
    facts_model = NDAFacts
    field_names = NDAFacts.FIELD_NAMES
    set_fields = NDAFacts.SET_FIELDS
    judgment_fields = NDAFacts.JUDGMENT_FIELDS
    extraction_prompt = EXTRACTION_PROMPT

    def __init__(self) -> None:
        self.rule_extractor = _rules.NDARuleExtractor()

    def field_risk(self) -> dict[str, str]:
        # How long secrets stay protected and whether materials come back is the NDA's
        # whole point; parties/jurisdiction are checkable at a glance; dates low-stakes.
        return {
            "return_of_materials": "high", "confidentiality_period": "high", "term": "high",
            "disclosing_party": "medium", "receiving_party": "medium", "governing_law": "medium",
            "effective_date": "low",
        }

    def classify_clause(self, chunk: Chunk) -> str:
        return _enrich.classify_clause(chunk)

    def permission_tags(self, chunk: Chunk) -> list[str]:
        return _enrich.permission_tags(chunk)

    def normalize_gold(self, raw: Mapping[str, str]) -> dict[str, str]:
        return _gold.normalize_facts(dict(raw))

    def canonicalize_value(self, name: str, value: str) -> str:
        return _SCALAR_CANON.get(name, lambda v: v)(value)

    def entities(self, value: str) -> list[str]:
        return party_entities(value)

    def empty_facts(self) -> NDAFacts:
        return NDAFacts()
