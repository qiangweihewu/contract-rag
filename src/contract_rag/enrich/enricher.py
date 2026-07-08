"""Generic enrichment engine. The clause taxonomy + ABAC tag rules are the active
vertical's; classify_clause/permission_tags are re-exported from the contract vertical
for back-compat."""
from __future__ import annotations

from typing import TYPE_CHECKING

from contract_rag.chunk.models import Chunk

if TYPE_CHECKING:
    from contract_rag.verticals.base import Vertical
from contract_rag.verticals.contract.enrich import (  # back-compat re-export
    classify_clause,
    permission_tags,
)

__all__ = ["classify_clause", "permission_tags", "enrich_chunks"]


def enrich_chunks(chunks: list[Chunk], vertical: Vertical | None = None) -> list[Chunk]:
    from contract_rag.verticals.registry import default_vertical

    v = vertical or default_vertical()
    out: list[Chunk] = []
    for c in chunks:
        c = c.model_copy(update={"clause_type": v.classify_clause(c)})
        c = c.model_copy(update={"permission_tags": v.permission_tags(c)})
        out.append(c)
    return out
