"""Dual-engine value-level omission cross-check (spec 2026-07-14).

A verifier engine's parse (e.g. dots.ocr — measured 0/72 number-fact omission)
is diffed against the primary parse (e.g. paddleocr) at the level of
canonicalized *critical tokens*: tokens carrying at least one digit (numbers,
dates, amounts, percentages) — the classes FinCriticalED showed get silently
dropped. A critical token the verifier read that the primary never emitted
anywhere is an omission candidate; `min_missing` of them flag the document.
Pure IR -> report; no I/O, no model calls. Pure-alpha tokens are excluded by
design: cross-checking prose drowns the signal in OCR spelling variance."""
from __future__ import annotations

from pydantic import BaseModel, Field

from contract_rag.clean.quality import QualityReport
from contract_rag.ir import DocumentIR


def _canon(text: str) -> str:
    from contract_rag.eval.fincritical import canon_fact_text

    return canon_fact_text(text)


def critical_tokens(text: str) -> set[str]:
    """Canonicalized tokens carrying >=1 digit (two-sided fincritical
    discipline: formatting never counts, digits/decimal/sign/% do)."""
    return {t for t in _canon(text).split() if any(c.isdigit() for c in t)}


class CrosscheckReport(BaseModel):
    missing_tokens: list[str] = Field(default_factory=list)
    missing_count: int = 0
    flagged: bool = False
    verifier_engine: str | None = None


def _ir_text(ir: DocumentIR) -> str:
    return " ".join(b.text for b in ir.blocks)


def crosscheck(
    primary_ir: DocumentIR, verifier_ir: DocumentIR, *, min_missing: int = 1
) -> CrosscheckReport:
    primary_tokens = set(_canon(_ir_text(primary_ir)).split())
    missing = sorted(critical_tokens(_ir_text(verifier_ir)) - primary_tokens)
    return CrosscheckReport(
        missing_tokens=missing,
        missing_count=len(missing),
        flagged=len(missing) >= min_missing,
        verifier_engine=(
            verifier_ir.blocks[0].source_engine if verifier_ir.blocks else None
        ),
    )


def annotate_report(cc: CrosscheckReport, report: QualityReport) -> QualityReport:
    """Additive only: `quality_score`/`needs_review` byte-identical — the same
    contract as the coverage/layout annotate seams."""
    return report.model_copy(update={
        "crosscheck_missing_count": cc.missing_count,
        "crosscheck_flagged": cc.flagged,
    })
