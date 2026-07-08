"""Pure pipeline-service logic behind the API. Kept free of FastAPI so it is
unit-testable with hand-built IR; api/app.py is a thin shell over these."""
from __future__ import annotations

from pydantic import BaseModel

from contract_rag.clean.pipeline import clean_ir
from contract_rag.clean.quality import QualityReport, compute_quality_score
from contract_rag.ir import DocumentIR


class Problem(BaseModel):
    code: str
    message: str
    severity: str  # "info" | "warn" | "error"


class Diagnosis(BaseModel):
    doc_id: str
    raw_quality: QualityReport
    cleaned_quality: QualityReport
    delta: float
    problems: list[Problem]
    redactions: int = 0


def quality_problems(report: QualityReport) -> list[Problem]:
    """Plain-English problems derived from a QualityReport — what makes a
    customer's RAG return garbage today."""
    problems: list[Problem] = []
    if report.garble_ratio > 0.05:
        problems.append(Problem(
            code="garble", severity="error",
            message=f"{report.garble_ratio:.0%} of blocks contain mojibake / encoding damage.",
        ))
    if report.empty_ratio > 0.1:
        problems.append(Problem(
            code="empty", severity="warn",
            message=f"{report.empty_ratio:.0%} of blocks are empty or whitespace-only.",
        ))
    if report.table_integrity < 0.99:
        problems.append(Problem(
            code="tables", severity="warn",
            message=f"Table integrity is {report.table_integrity:.0%}; tables look broken.",
        ))
    if report.mean_confidence < 0.75:
        problems.append(Problem(
            code="confidence", severity="warn",
            message=f"Mean OCR/parse confidence is {report.mean_confidence:.0%}.",
        ))
    return problems


def diagnose_ir(ir: DocumentIR, *, doc_id: str | None = None, redactions: int = 0) -> Diagnosis:
    raw = compute_quality_score(ir)
    cleaned = compute_quality_score(clean_ir(ir))
    return Diagnosis(
        doc_id=doc_id or ir.doc_id,
        raw_quality=raw,
        cleaned_quality=cleaned,
        delta=round(cleaned.quality_score - raw.quality_score, 3),
        problems=quality_problems(raw),
        redactions=redactions,
    )
