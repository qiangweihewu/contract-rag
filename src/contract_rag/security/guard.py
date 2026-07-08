"""Defense-in-depth permission guard: even after the retriever filters by
allowed_tags, audit the returned set so any leak is caught and surfaced."""
from __future__ import annotations

from pydantic import BaseModel

from contract_rag.chunk.models import Chunk


class Violation(BaseModel):
    chunk_id: str
    tags: list[str]
    reason: str


def permitted(chunk: Chunk, allowed_tags: list[str]) -> bool:
    return bool(set(chunk.permission_tags) & set(allowed_tags))


def audit_results(chunks: list[Chunk], allowed_tags: list[str]) -> list[Violation]:
    return [
        Violation(chunk_id=c.chunk_id, tags=c.permission_tags, reason="tag_not_permitted")
        for c in chunks
        if not permitted(c, allowed_tags)
    ]
