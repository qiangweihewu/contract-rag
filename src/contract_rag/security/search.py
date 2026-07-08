"""Identity-driven retrieval: principal -> allowed_tags -> index.search -> audit.
Composes the existing HybridIndex.search(allowed_tags=...) filter with a guard."""
from __future__ import annotations

from contract_rag.chunk.models import Chunk
from contract_rag.security.abac import Principal, allowed_tags_for
from contract_rag.security.guard import audit_results


def search_as(index, query: str, principal: Principal, k: int = 5, **kwargs) -> list[Chunk]:
    allowed = allowed_tags_for(principal)
    results = index.search(query, k=k, allowed_tags=allowed, **kwargs)
    violations = audit_results(results, allowed)
    if violations:
        raise PermissionError(
            f"retrieval leak for subject={principal.subject}: "
            f"{[v.chunk_id for v in violations]} not in allowed_tags={allowed}"
        )
    return results
