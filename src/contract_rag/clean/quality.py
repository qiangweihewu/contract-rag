from __future__ import annotations

from pydantic import BaseModel, Field

from contract_rag.ir import BlockType, DocumentIR

_MOJIBAKE_MARKERS = ("Ã", "Â", "â€", "Ã¢", "â‚¬")


class QualityReport(BaseModel):
    quality_score: float
    garble_ratio: float
    empty_ratio: float
    table_integrity: float
    mean_confidence: float
    needs_review: bool
    breakdown: dict = Field(default_factory=dict)
    # Omission-aware geometric signal (clean/coverage.py), populated ONLY on the
    # scanned/paddle path when a rendered page + bboxes are available; None on the
    # text-only path. Additive: it never enters the `quality_score` computation, so
    # scores are byte-identical whether or not coverage is attached.
    ink_coverage: float | None = None
    uncovered_ink_ratio: float | None = None


def is_garbled(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if "�" in t:  # Unicode replacement char = decode failure
        return True
    if any(m in t for m in _MOJIBAKE_MARKERS):
        return True
    non_ascii = sum(ord(c) > 0x7F for c in t)
    return non_ascii / len(t) > 0.15


def table_integrity(ir: DocumentIR) -> float:
    tables = [b for b in ir.blocks if b.type is BlockType.TABLE]
    if not tables:
        return 1.0
    intact = sum(1 for b in tables if ("|" in b.text or "\n" in b.text))
    return intact / len(tables)


def compute_quality_score(ir: DocumentIR) -> QualityReport:
    blocks = ir.blocks
    n = len(blocks)
    if n == 0:
        return QualityReport(
            quality_score=0.0, garble_ratio=0.0, empty_ratio=1.0,
            table_integrity=1.0, mean_confidence=0.0, needs_review=True,
            breakdown={"n_blocks": 0},
        )
    garble = sum(is_garbled(b.text) for b in blocks) / n
    empty = sum(not b.text.strip() for b in blocks) / n
    tbl = table_integrity(ir)
    mean_conf = sum(b.confidence for b in blocks) / n
    score = 0.45 * (1 - garble) + 0.25 * tbl + 0.15 * (1 - empty) + 0.15 * mean_conf
    return QualityReport(
        quality_score=round(score, 3),
        garble_ratio=round(garble, 3),
        empty_ratio=round(empty, 3),
        table_integrity=round(tbl, 3),
        mean_confidence=round(mean_conf, 3),
        needs_review=score < 0.75,
        breakdown={"n_blocks": n},
    )
