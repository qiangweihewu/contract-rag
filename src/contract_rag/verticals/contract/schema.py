from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from contract_rag.verticals.base import ExtractedClause  # generic primitive lives in base


class ContractFacts(BaseModel):
    counterparty: ExtractedClause
    effective_date: ExtractedClause
    governing_law: ExtractedClause
    # Phase-4 fields default to empty so 3-field construction stays valid.
    total_value: ExtractedClause = Field(default_factory=ExtractedClause)
    termination_notice_days: ExtractedClause = Field(default_factory=ExtractedClause)
    auto_renewal: ExtractedClause = Field(default_factory=ExtractedClause)

    FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "counterparty", "effective_date", "governing_law",
        "total_value", "termination_notice_days", "auto_renewal",
    )
    # Multi-valued fields: compared/verified by entity-set overlap, not scalar equality.
    SET_FIELDS: ClassVar[tuple[str, ...]] = ("counterparty",)
    # Derived judgments (not verbatim spans): exempt from span source-attribution.
    JUDGMENT_FIELDS: ClassVar[tuple[str, ...]] = ("auto_renewal",)


__all__ = ["ContractFacts", "ExtractedClause"]
