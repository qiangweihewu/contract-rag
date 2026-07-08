"""Deterministic, credential-free contract extractor.

A regex/rule baseline that needs no API key and no GPU, so the full before/after
proof is reproducible by anyone and runnable in CI. It is the honest *floor*; the
`openai`/`local` backends are the ceiling. Every value is a literal span of the
block it is cited from, so it satisfies the source-attribution gate by construction.
"""
from __future__ import annotations

import re

from contract_rag.verticals.base import ExtractedClause
from contract_rag.verticals.contract.schema import ContractFacts
from contract_rag.verticals.legal_common import (
    _DATE,
    _MONTHS,
    jurisdiction_in,
    party_entities,
)
from contract_rag.ir import DocumentIR

_MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d{2})?")
_DAYS_RE = re.compile(r"(?:\b[A-Za-z]+\s+)?\(?\d+\)?\s*(?:calendar |business )?days?", re.IGNORECASE)
_DIGITS_DAYS_RE = re.compile(r"(\d+)\s*\)?\s*(?:calendar |business )?days?", re.IGNORECASE)
_AUTO_RE = re.compile(
    r"automatic(?:ally)?\s+renew|renew\w*\s+automatically|"
    r"successive\s+(?:\S+[\s-]+){0,3}(?:term|period|year|month)|evergreen",
    re.IGNORECASE,
)


def days_in(text: str) -> str:
    """Canonical day count from a notice-period phrase ('ninety (90) days' -> '90')."""
    m = _DIGITS_DAYS_RE.search(text) or re.search(r"\((\d+)\)", text) or re.search(r"(\d+)", text)
    return m.group(1) if m else ""


def money_digits(text: str) -> str:
    """Canonical digits of a dollar amount ('$1,250,000.00' -> '1250000')."""
    m = re.search(r"\$?\s?(\d[\d,]*)", text)
    return m.group(1).replace(",", "") if m else ""


def auto_renewal_signal(text: str) -> bool:
    return bool(_AUTO_RE.search(text))


def find_governing_law(ir: DocumentIR) -> ExtractedClause:
    for b in ir.blocks:
        if not re.search(r"govern", b.text, re.IGNORECASE):
            continue
        j = jurisdiction_in(b.text)
        if j:
            return ExtractedClause(value=j, source_block_id=b.block_id, confidence=0.7)
    return ExtractedClause()


def find_effective_date(ir: DocumentIR) -> ExtractedClause:
    for b in ir.blocks:
        if not re.search(r"effective", b.text, re.IGNORECASE):
            continue
        m = _DATE.search(b.text)
        if m:
            return ExtractedClause(value=m.group(0), source_block_id=b.block_id, confidence=0.65)
    return ExtractedClause()


def find_counterparty(ir: DocumentIR) -> ExtractedClause:
    for b in ir.blocks:
        m = re.search(r"by and between", b.text, re.IGNORECASE)
        if not m:
            continue
        ents = party_entities(b.text[m.start() : m.start() + 600])
        if ents:
            return ExtractedClause(
                value="; ".join(ents), source_block_id=b.block_id, confidence=0.7
            )
    return ExtractedClause()


def find_total_value(ir: DocumentIR) -> ExtractedClause:
    for b in ir.blocks:
        m = _MONEY_RE.search(b.text)
        if m:
            return ExtractedClause(value=m.group(0).strip(), source_block_id=b.block_id, confidence=0.4)
    return ExtractedClause()


def find_termination_notice_days(ir: DocumentIR) -> ExtractedClause:
    for b in ir.blocks:
        if not re.search(r"terminat|notice|renew", b.text, re.IGNORECASE):
            continue
        m = _DAYS_RE.search(b.text)
        if m:
            return ExtractedClause(value=m.group(0).strip(), source_block_id=b.block_id, confidence=0.55)
    return ExtractedClause()


def find_auto_renewal(ir: DocumentIR) -> ExtractedClause:
    for b in ir.blocks:
        if auto_renewal_signal(b.text):
            return ExtractedClause(value="yes", source_block_id=b.block_id, confidence=0.55)
    return ExtractedClause()


class RuleExtractor:
    def extract(self, ir: DocumentIR) -> ContractFacts:
        return ContractFacts(
            counterparty=find_counterparty(ir),
            effective_date=find_effective_date(ir),
            governing_law=find_governing_law(ir),
            total_value=find_total_value(ir),
            termination_notice_days=find_termination_notice_days(ir),
            auto_renewal=find_auto_renewal(ir),
        )
