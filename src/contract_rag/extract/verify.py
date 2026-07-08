"""Hallucination guard (spec §5.4 guarantee #1).

Every extracted field must (a) actually appear in its cited block and (b) clear a
confidence floor, or it is quarantined to a human-in-the-loop queue and never
auto-written. This lets a high-recall extractor (e.g. an LLM) stay safe: it keeps
what is attributable and confident, and routes the rest to review.

The confidence floor is a flat constant by default (back-compat). Opt in to
per-risk-tier floors by passing `tier_thresholds={"high": ..., "medium": ...,
"low": ...}` — each field's floor is resolved via the vertical's optional
`field_risk` seam (fields default to "medium"; tiers missing from the map fall
back to the flat threshold).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import BaseModel

from contract_rag.ir import DocumentIR
from contract_rag.text import normalize

if TYPE_CHECKING:
    from contract_rag.verticals.base import Vertical

CONFIDENCE_THRESHOLD = 0.6


class FieldCheck(BaseModel):
    field: str
    value: str
    source_block_id: str | None
    confidence: float
    attributed: bool
    passed: bool
    reasons: list[str]


class VerificationReport(BaseModel):
    checks: dict[str, FieldCheck]

    @property
    def verified(self) -> dict[str, FieldCheck]:
        """Fields safe to auto-write."""
        return {k: c for k, c in self.checks.items() if c.passed}

    @property
    def quarantined(self) -> dict[str, FieldCheck]:
        """Non-empty extractions that failed a check → HITL queue."""
        return {k: c for k, c in self.checks.items() if c.value and not c.passed}


def _attributed(vertical, field: str, value: str, cited_block_text: str) -> bool:
    if field in vertical.set_fields:
        entities = [normalize(e) for e in vertical.entities(value)]
        return bool(entities) and all(e in cited_block_text for e in entities)
    return normalize(value) in cited_block_text


def resolve_thresholds(
    vertical, default: float, tier_thresholds: Mapping[str, float] | None
) -> dict[str, float]:
    """Per-field confidence floor. Without `tier_thresholds` every field gets the
    flat `default` (historical behavior, unchanged). With it, each field's floor is
    looked up by its risk tier (the same optional `field_risk` vertical seam the
    metrics use, resolved defensively — verticals without it are all "medium");
    tiers missing from the map fall back to `default`."""
    if tier_thresholds is None:
        return {name: default for name in vertical.field_names}
    from contract_rag.eval.metrics import field_risk_map

    risk = field_risk_map(vertical)
    return {
        name: float(tier_thresholds.get(risk[name], default))
        for name in vertical.field_names
    }


def verify(
    facts, ir: DocumentIR, threshold: float = CONFIDENCE_THRESHOLD,
    *, vertical: Vertical | None = None,
    tier_thresholds: Mapping[str, float] | None = None,
) -> VerificationReport:
    from contract_rag.verticals.registry import default_vertical

    v = vertical or default_vertical()
    floors = resolve_thresholds(v, threshold, tier_thresholds)
    block_text = {b.block_id: normalize(b.text) for b in ir.blocks}
    checks: dict[str, FieldCheck] = {}
    for name in v.field_names:
        clause = getattr(facts, name)
        if not clause.value:
            checks[name] = FieldCheck(
                field=name, value="", source_block_id=clause.source_block_id,
                confidence=clause.confidence, attributed=False, passed=False, reasons=["empty"],
            )
            continue
        cited = block_text.get(clause.source_block_id or "", "")
        # judgment fields (e.g. auto_renewal "yes") have no verbatim span to attribute.
        attributed = True if name in v.judgment_fields else _attributed(v, name, clause.value, cited)
        reasons: list[str] = []
        if not attributed:
            reasons.append("unattributed")
        if clause.confidence < floors[name]:
            reasons.append("low_confidence")
        checks[name] = FieldCheck(
            field=name, value=clause.value, source_block_id=clause.source_block_id,
            confidence=clause.confidence, attributed=attributed,
            passed=not reasons, reasons=reasons,
        )
    return VerificationReport(checks=checks)
