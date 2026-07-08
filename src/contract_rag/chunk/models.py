from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    block_ids: list[str]              # source blocks — keeps attribution through retrieval
    heading: str | None = None        # nearest enclosing heading (parent context)
    page: int | None = None
    clause_type: str | None = None    # set by enrich/
    permission_tags: list[str] = Field(default_factory=list)  # ABAC, set by enrich/
    metadata: dict = Field(default_factory=dict)
    index_extra: str = ""             # extra text for retrieval indexing only (never displayed)
    definition_block_ids: list[str] = Field(default_factory=list)  # provenance of injected defs

    def index_text(self) -> str:
        """Text handed to retrievers for indexing/embedding. Kept separate from `text`
        (the display value) so injected content (e.g. definitions) never leaks into the
        UI. With no `index_extra` this is byte-identical to the pre-injection format."""
        if self.index_extra:
            return f"{self.heading or ''} {self.index_extra} {self.text}"
        return f"{self.heading or ''} {self.text}"
