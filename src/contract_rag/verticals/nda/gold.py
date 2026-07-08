"""Canonicalize NDA golden labels to the extractor's answer space — the same helpers
both sides use, so the metric measures the answer, not the phrasing."""
from __future__ import annotations

import re

from contract_rag.verticals.legal_common import _DATE, jurisdiction_in, party_entities
from contract_rag.verticals.nda.rules import duration_in


def _entities(raw: str) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for e in party_entities(raw):
        if e.lower() not in seen:
            seen.add(e.lower())
            out.append(e)
    return "; ".join(out)


def _date(raw: str) -> str:
    m = _DATE.search(raw)
    return m.group(0).strip() if m else raw.strip()


def _yesno(raw: str) -> str:
    t = raw.strip().lower()
    if not t:
        return ""
    if re.search(r"\byes\b|\btrue\b|return|destroy", t):
        return "yes"
    if re.search(r"\bno\b|\bfalse\b", t):
        return "no"
    return "no"


_FACT_NORMALIZERS = {
    "disclosing_party": _entities,
    "receiving_party": _entities,
    "effective_date": _date,
    "term": duration_in,
    "confidentiality_period": duration_in,
    "return_of_materials": _yesno,
    "governing_law": lambda v: jurisdiction_in(v) or "",
}


def normalize_facts(facts: dict[str, str]) -> dict[str, str]:
    return {k: _FACT_NORMALIZERS.get(k, lambda x: x)(v) for k, v in facts.items()}
