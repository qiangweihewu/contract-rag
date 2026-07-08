"""LLM pre-label + human-correct loop (F1): run any Extractor to DRAFT a GoldenDoc,
capture human corrections as an immutable overlay, emit an APPROVED GoldenDoc in the
extractor's answer space (normalize_facts). Default extractor is the credential-free
`rule` backend, so the loop runs with no secrets."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from contract_rag.verticals.contract.gold import normalize_facts
from contract_rag.eval.golden import GoldenDoc
from contract_rag.extract.schema import ContractFacts
from contract_rag.ir import DocumentIR


class PrelabelStatus(str, Enum):
    DRAFT = "draft"
    CORRECTED = "corrected"
    APPROVED = "approved"


class PrelabelRecord(BaseModel):
    doc_id: str
    source_pdf: str
    draft_facts: dict[str, str]
    model: str
    status: PrelabelStatus = PrelabelStatus.DRAFT
    corrections: dict[str, str] = Field(default_factory=dict)


def prelabel(
    doc_id: str, source_pdf: str, ir: DocumentIR, extractor, model: str = "rule"
) -> PrelabelRecord:
    facts = extractor.extract(ir)
    draft = {name: getattr(facts, name).value for name in ContractFacts.FIELD_NAMES}
    return PrelabelRecord(doc_id=doc_id, source_pdf=source_pdf, draft_facts=draft, model=model)


def apply_corrections(record: PrelabelRecord, corrections: dict[str, str]) -> PrelabelRecord:
    unknown = set(corrections) - set(ContractFacts.FIELD_NAMES)
    if unknown:
        raise ValueError(
            f"unknown correction field(s) {sorted(unknown)}; "
            f"valid fields are {list(ContractFacts.FIELD_NAMES)}"
        )
    merged = {**record.corrections, **corrections}
    return record.model_copy(update={"corrections": merged, "status": PrelabelStatus.CORRECTED})


def approve(record: PrelabelRecord) -> PrelabelRecord:
    return record.model_copy(update={"status": PrelabelStatus.APPROVED})


def to_golden(record: PrelabelRecord) -> GoldenDoc:
    facts = normalize_facts({**record.draft_facts, **record.corrections})
    # Fail loud on silent gold corruption: a human correction is the most authoritative
    # signal there is, so if canonicalization empties it (e.g. governing_law="NY" can't be
    # represented as a jurisdiction), surface it instead of writing an empty gold field.
    dropped = [f for f, v in record.corrections.items() if v and not facts.get(f)]
    if dropped:
        raise ValueError(
            f"correction(s) {dropped} normalized to empty in the extractor's answer space; "
            f"the value cannot be represented as gold for this field — fix the correction or "
            f"add the form to the canonicalizer (eval/cuad.py)"
        )
    return GoldenDoc(doc_id=record.doc_id, source_pdf=record.source_pdf, facts=facts)
