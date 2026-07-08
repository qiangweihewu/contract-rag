from __future__ import annotations

import re
from collections.abc import Mapping

from contract_rag.chunk.models import Chunk
from contract_rag.verticals.base import ExtractedClause
from contract_rag.verticals.contract import enrich as _enrich
from contract_rag.verticals.contract import gold as _gold
from contract_rag.verticals.contract import rules as _rules
from contract_rag.verticals.contract.prompt import EXTRACTION_PROMPT
from contract_rag.verticals.contract.schema import ContractFacts


def _yesno(v: str) -> str:
    t = v.strip().lower()
    if not t:
        return ""
    if re.search(r"\byes\b|\btrue\b", t) or _rules.auto_renewal_signal(v):
        return "yes"
    if re.search(r"\bno\b|\bfalse\b", t):
        return "no"
    return "no"


# Scalar fields whose surface form varies by backend are canonicalized on BOTH sides so
# the metric measures the answer, not the phrasing.
_SCALAR_CANON = {
    "governing_law": lambda v: _rules.jurisdiction_in(v) or "",
    "total_value": _rules.money_digits,
    "termination_notice_days": _rules.days_in,
    "auto_renewal": _yesno,
}


class ContractVertical:
    name = "contract"
    facts_model = ContractFacts
    field_names = ContractFacts.FIELD_NAMES
    set_fields = ContractFacts.SET_FIELDS
    judgment_fields = ContractFacts.JUDGMENT_FIELDS
    extraction_prompt = EXTRACTION_PROMPT

    def __init__(self) -> None:
        self.rule_extractor = _rules.RuleExtractor()

    def field_risk(self) -> dict[str, str]:
        # Money, termination and renewal terms are where extraction errors cost the
        # customer; parties/jurisdiction are checkable at a glance; dates are low-stakes.
        return {
            "total_value": "high", "termination_notice_days": "high", "auto_renewal": "high",
            "counterparty": "medium", "governing_law": "medium",
            "effective_date": "low",
        }

    def classify_clause(self, chunk: Chunk) -> str:
        return _enrich.classify_clause(chunk)

    def permission_tags(self, chunk: Chunk) -> list[str]:
        return _enrich.permission_tags(chunk)

    def normalize_gold(self, raw: Mapping[str, str]) -> dict[str, str]:
        return _gold.normalize_facts(dict(raw))

    def canonicalize_value(self, name: str, value: str) -> str:
        return _SCALAR_CANON.get(name, lambda x: x)(value)

    def entities(self, value: str) -> list[str]:
        return _rules.party_entities(value)

    def empty_facts(self) -> ContractFacts:
        return ContractFacts(
            counterparty=ExtractedClause(),
            effective_date=ExtractedClause(),
            governing_law=ExtractedClause(),
        )
