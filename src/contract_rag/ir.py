from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class BlockType(str, Enum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST_ITEM = "list_item"
    FOOTER = "footer"
    HEADER = "header"
    FIGURE_CAPTION = "figure_caption"


class BoundingBox(BaseModel):
    page: int
    x0: float
    y0: float
    x1: float
    y1: float


class DocBlock(BaseModel):
    block_id: str
    type: BlockType
    text: str
    bbox: BoundingBox | None = None
    parent_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source_engine: str


class DocumentIR(BaseModel):
    doc_id: str
    source_uri: str
    file_hash: str
    mime_type: str
    blocks: list[DocBlock]
    metadata: dict = Field(default_factory=dict)
