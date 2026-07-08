"""Deterministic, credential-free NDA extractor (vertical #2).

Mirrors the contract `rule` extractor: every value is a verbatim span of its cited
block, so source-attribution holds by construction. Reuses the general legal
primitives (jurisdiction, party entities, dates) from the contract vertical; the
duration and return-of-materials finders are NDA-specific.
"""
from __future__ import annotations

import re

from contract_rag.ir import DocumentIR
from contract_rag.verticals.base import ExtractedClause
from contract_rag.verticals.legal_common import _MONTHS, jurisdiction_in, party_entities
from contract_rag.verticals.nda.schema import NDAFacts

# "two (2) years", "thirty-six (36) months", "5 years", and word-number forms
# ("three years from the date of this letter"). The (?<!-) keeps compound word
# numbers ("thirty-one years") from half-matching as "one years".
_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_DURATION_RE = re.compile(
    rf"\(?(\d+)\)?\s*(year|month)s?"
    rf"|\b(?<!-)({'|'.join(_WORD_NUM)})\s+(year|month)s?",
    re.IGNORECASE,
)
_RETURN_RE = re.compile(
    r"return\s+or\s+destroy|return\s+and\s+destroy|return\s+all|promptly\s+destroy|destroy\s+all",
    re.IGNORECASE,
)
# defined-term role labels: nearest party entity BEFORE the label is that role's party
_DISCLOSING_RE = re.compile(r"disclos\w*\s+party|\bdiscloser\b", re.IGNORECASE)
_RECEIVING_RE = re.compile(r"receiv\w*\s+party|\brecipient\b", re.IGNORECASE)
# preamble cue for the party fallback ("by and among" covers multi-party NDAs)
_PREAMBLE_RE = re.compile(r"by and between|by and among|between", re.IGNORECASE)
_HEAD_SCAN_BLOCKS = 20  # "PARTIES:"-list / letterhead styles sit at the top of the doc

# Date forms real SEC NDAs use (a superset of the contract vertical's _DATE):
# legalese ordinals ("the 6th day of January, 2012" / "4th day of May 2005"),
# month + ordinal day ("April 6th, 2005"), day-month-year ("6 January 2012"),
# plus the plain prose/slash/ISO forms. \s covers the nbsp these filings carry.
_ORD = r"(?:st|nd|rd|th)?"
_NDA_DATE = re.compile(
    rf"\d{{1,2}}{_ORD}\s+day\s+of\s+(?:{_MONTHS})[\s,]*\d{{4}}"
    rf"|(?:{_MONTHS})\s+\d{{1,2}}{_ORD}\s*,?\s*\d{{4}}"
    rf"|\d{{1,2}}{_ORD}\s+(?:{_MONTHS})\s*,?\s*\d{{4}}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|\d{4}-\d{1,2}-\d{1,2}",
    re.IGNORECASE,
)
# agreement-date cue that must appear shortly BEFORE the date ("dated this ...",
# "is made as of ...", "executed this ..."); block-level cues alone pick up dates
# from unrelated clauses that merely mention "effective"
_DATE_CUE = re.compile(
    r"\b(?:effective|dated?|made|entered\s+into|executed|as\s+of)\b", re.IGNORECASE
)
_CUE_WINDOW = 80


def duration_in(text: str) -> str:
    """Canonical duration ('two (2) years' -> '2 years', 'three years' -> '3 years')."""
    m = _DURATION_RE.search(text)
    if not m:
        return ""
    if m.group(1):
        return f"{m.group(1)} {m.group(2).lower()}s"
    return f"{_WORD_NUM[m.group(3).lower()]} {m.group(4).lower()}s"


def return_signal(text: str) -> bool:
    return bool(_RETURN_RE.search(text))


def _party_before(text: str, label_re: re.Pattern[str]) -> list[str]:
    """Entities appearing in the ~250 chars before the first role-label match (nearest last)."""
    m = label_re.search(text)
    if not m:
        return []
    ents = party_entities(text[max(0, m.start() - 250): m.start()])
    return ents[-1:] if ents else []


def _find_party(ir: DocumentIR, label_re: re.Pattern[str]) -> ExtractedClause:
    for b in ir.blocks:
        if not re.search(r"by and between|between", b.text, re.IGNORECASE):
            continue
        ents = _party_before(b.text, label_re)
        if ents:
            return ExtractedClause(value="; ".join(ents), source_block_id=b.block_id, confidence=0.7)
    return ExtractedClause()


def find_disclosing_party(ir: DocumentIR) -> ExtractedClause:
    return _find_party(ir, _DISCLOSING_RE)


def find_receiving_party(ir: DocumentIR) -> ExtractedClause:
    return _find_party(ir, _RECEIVING_RE)


def _preamble_parties(ir: DocumentIR) -> tuple[list[str], str]:
    """Party entities from the preamble, for when role labels don't fire.

    Tier 1: the first block with a 'by and between/among' style cue that yields
    corporate entities in the 600 chars after the cue (the preamble party list).
    Tier 2: 'PARTIES:'-list / letterhead styles have no 'between' at all — take the
    first head block that yields any entity. Returns (entities, block_id)."""
    for b in ir.blocks:
        m = _PREAMBLE_RE.search(b.text)
        if not m:
            continue
        ents = party_entities(b.text[m.start(): m.start() + 600])
        if ents:
            return ents, b.block_id
    for b in ir.blocks[:_HEAD_SCAN_BLOCKS]:
        ents = party_entities(b.text)
        if ents:
            return ents, b.block_id
    return [], ""


def _assign_parties(ir: DocumentIR) -> tuple[ExtractedClause, ExtractedClause]:
    """Disclosing/receiving with a preamble fallback. Role labels always win; the
    fallback only fills roles the labels left empty.

    Role assignment from an unlabeled preamble is genuinely ambiguous (mutual NDAs
    disclose both ways). Convention chosen: the FIRST-named entity is the disclosing
    party, the rest are receiving — NDA/employment preambles conventionally name the
    protagonist whose information is being protected first (all 8 synthetic
    examples/nda docs and the SEC employment NDAs follow it: employer/discloser
    first, employee/recipient second). Confidence 0.5 (vs the label heuristic's 0.7)
    marks the assignment as heuristic. Both roles cite the one preamble block, so
    verify()'s span attribution holds."""
    disclosing = find_disclosing_party(ir)
    receiving = find_receiving_party(ir)
    if disclosing.value and receiving.value:
        return disclosing, receiving
    ents, block_id = _preamble_parties(ir)
    taken = {e.lower() for e in party_entities(disclosing.value + "; " + receiving.value)}
    rest = [e for e in ents if e.lower() not in taken]
    if not rest:
        return disclosing, receiving
    if not disclosing.value and not receiving.value:
        disclosing = ExtractedClause(value=rest[0], source_block_id=block_id, confidence=0.5)
        if rest[1:]:
            receiving = ExtractedClause(
                value="; ".join(rest[1:]), source_block_id=block_id, confidence=0.5)
    elif not disclosing.value:
        disclosing = ExtractedClause(
            value="; ".join(rest), source_block_id=block_id, confidence=0.5)
    else:
        receiving = ExtractedClause(
            value="; ".join(rest), source_block_id=block_id, confidence=0.5)
    return disclosing, receiving


def find_effective_date(ir: DocumentIR) -> ExtractedClause:
    # Pass 1: first date (doc order) whose preceding window carries an
    # agreement-date cue — the preamble "dated/made/entered into as of <date>".
    for b in ir.blocks:
        for m in _NDA_DATE.finditer(b.text):
            if _DATE_CUE.search(b.text[max(0, m.start() - _CUE_WINDOW): m.start()]):
                return ExtractedClause(
                    value=m.group(0).strip(), source_block_id=b.block_id, confidence=0.65)
    # Pass 2: letter-style NDAs date a standalone line ("December 11,2014",
    # "Date: 5/12/09") with no cue; accept a block that is essentially just a date.
    for b in ir.blocks:
        m = _NDA_DATE.search(b.text)
        if m and len(re.sub(r"[\W_]+", "", b.text[: m.start()] + b.text[m.end():])) <= 8:
            return ExtractedClause(
                value=m.group(0).strip(), source_block_id=b.block_id, confidence=0.4)
    return ExtractedClause()


def _find_duration(ir: DocumentIR, cue: str) -> ExtractedClause:
    for b in ir.blocks:
        if not re.search(cue, b.text, re.IGNORECASE):
            continue
        m = _DURATION_RE.search(b.text)
        if m:
            return ExtractedClause(value=m.group(0).strip(), source_block_id=b.block_id, confidence=0.6)
    return ExtractedClause()


def find_term(ir: DocumentIR) -> ExtractedClause:
    # "shall terminate <n> years after/from ..." is how SEC NDAs' Term clauses read
    return _find_duration(
        ir,
        r"remain in (?:full )?(?:force|effect)|term of this|continue for|"
        r"for a (?:term|period) of|shall terminate",
    )


def find_confidentiality_period(ir: DocumentIR) -> ExtractedClause:
    return _find_duration(ir, r"survive|obligations? of confidentiality|confidential\w*.*(?:survive|continue|remain)")


def find_return_of_materials(ir: DocumentIR) -> ExtractedClause:
    for b in ir.blocks:
        if return_signal(b.text):
            return ExtractedClause(value="yes", source_block_id=b.block_id, confidence=0.6)
    return ExtractedClause()


def find_governing_law(ir: DocumentIR) -> ExtractedClause:
    for b in ir.blocks:
        if not re.search(r"govern", b.text, re.IGNORECASE):
            continue
        j = jurisdiction_in(b.text)
        if j:
            return ExtractedClause(value=j, source_block_id=b.block_id, confidence=0.7)
    return ExtractedClause()


class NDARuleExtractor:
    def extract(self, ir: DocumentIR) -> NDAFacts:
        disclosing, receiving = _assign_parties(ir)
        return NDAFacts(
            disclosing_party=disclosing,
            receiving_party=receiving,
            effective_date=find_effective_date(ir),
            term=find_term(ir),
            confidentiality_period=find_confidentiality_period(ir),
            return_of_materials=find_return_of_materials(ir),
            governing_law=find_governing_law(ir),
        )
