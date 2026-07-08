from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from contract_rag.chunk.models import Chunk


class ExtractedClause(BaseModel):
    value: str = Field(default="", description="extracted field value; empty if not found")
    source_block_id: str | None = Field(
        default=None, description="block_id the value came from; required when value is non-empty"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


@runtime_checkable
class Vertical(Protocol):
    """Everything a vertical must supply so the generic engines (extract, enrich,
    metrics, verify, eval) need no per-vertical edits. Adding a vertical = implement
    this + register it; no core fork.

    Optional seam (deliberately NOT part of the protocol, so existing third-party
    verticals stay valid): `field_risk` — a mapping or zero-arg method returning
    {field: "high"|"medium"|"low"}. Resolved defensively by
    eval.metrics.field_risk_map(); missing fields/levels default to "medium"."""

    name: str
    facts_model: type[BaseModel]      # facts schema; every field an ExtractedClause
    field_names: tuple[str, ...]      # drives extraction prompts + metrics
    set_fields: tuple[str, ...]       # multi-valued: entity-set overlap, not scalar eq
    judgment_fields: tuple[str, ...]  # derived judgments, exempt from span attribution
    extraction_prompt: str            # per-field instructions for instructor backends
    rule_extractor: object            # an Extractor: extract(ir) -> facts_model

    def classify_clause(self, chunk: Chunk) -> str: ...
    def permission_tags(self, chunk: Chunk) -> list[str]: ...
    def normalize_gold(self, raw: Mapping[str, str]) -> dict[str, str]: ...
    def canonicalize_value(self, name: str, value: str) -> str: ...  # scalar match canon
    def entities(self, value: str) -> list[str]: ...                 # set-field members
    def empty_facts(self) -> BaseModel: ...
