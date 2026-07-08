from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from contract_rag.verticals.base import ExtractedClause


class NDAFacts(BaseModel):
    disclosing_party: ExtractedClause = Field(default_factory=ExtractedClause)
    receiving_party: ExtractedClause = Field(default_factory=ExtractedClause)
    effective_date: ExtractedClause = Field(default_factory=ExtractedClause)
    term: ExtractedClause = Field(default_factory=ExtractedClause)
    confidentiality_period: ExtractedClause = Field(default_factory=ExtractedClause)
    return_of_materials: ExtractedClause = Field(default_factory=ExtractedClause)
    governing_law: ExtractedClause = Field(default_factory=ExtractedClause)

    FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "disclosing_party", "receiving_party", "effective_date", "term",
        "confidentiality_period", "return_of_materials", "governing_law",
    )
    # Multi-valued fields: compared/verified by entity-set overlap, not scalar equality.
    SET_FIELDS: ClassVar[tuple[str, ...]] = ("disclosing_party", "receiving_party")
    # Derived judgments (not verbatim spans): exempt from span source-attribution.
    JUDGMENT_FIELDS: ClassVar[tuple[str, ...]] = ("return_of_materials",)


__all__ = ["NDAFacts"]
