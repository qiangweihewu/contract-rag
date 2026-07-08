from __future__ import annotations

import ast

from contract_rag.verticals.contract.rules import (
    _DATE,
    auto_renewal_signal,
    days_in,
    jurisdiction_in,
    party_entities,
)


def _parse_span_list(raw: str) -> list[str]:
    """CUAD answers are stringified Python lists of highlighted spans; recover the spans."""
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        val = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return [raw]
    return [str(x) for x in val] if isinstance(val, list) else [str(val)]


def normalize_governing_law(raw: str) -> str:
    """Reduce a CUAD governing-law span (a whole clause) to its jurisdiction scalar."""
    for span in _parse_span_list(raw):
        j = jurisdiction_in(span)
        if j:
            return j
    return ""


def normalize_effective_date(raw: str) -> str:
    """Reduce a CUAD effective-date span to a bare date; empty if the span is prose."""
    for span in _parse_span_list(raw):
        m = _DATE.search(span)
        if m:
            return m.group(0).strip()
    return ""


def normalize_counterparty(raw: str) -> str:
    """Reduce CUAD's party span-list (legal names mixed with defined-term aliases) to
    the set of corporate entity names, "; "-joined. Metrics compare this as a set."""
    out: list[str] = []
    seen: set[str] = set()
    for span in _parse_span_list(raw):
        for e in party_entities(span):
            key = e.lower()
            if key not in seen:
                seen.add(key)
                out.append(e)
    return "; ".join(out)


def normalize_termination_notice_days(raw: str) -> str:
    """CUAD notice-period span -> integer day count ('ninety (90) days' -> '90')."""
    for span in _parse_span_list(raw):
        d = days_in(span)
        if d:
            return d
    return ""


def normalize_auto_renewal(raw: str) -> str:
    """CUAD 'Renewal Term' span -> 'yes' if it describes automatic renewal, else 'no'
    when a renewal term exists, '' when none is labeled."""
    spans = _parse_span_list(raw)
    if not spans:
        return ""
    return "yes" if auto_renewal_signal(" ".join(spans)) else "no"


_FACT_NORMALIZERS = {
    "counterparty": normalize_counterparty,
    "effective_date": normalize_effective_date,
    "governing_law": normalize_governing_law,
    "termination_notice_days": normalize_termination_notice_days,
    "auto_renewal": normalize_auto_renewal,
}


def normalize_facts(facts: dict[str, str]) -> dict[str, str]:
    return {k: _FACT_NORMALIZERS.get(k, lambda x: x)(v) for k, v in facts.items()}
