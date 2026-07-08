from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


class Span(BaseModel):
    name: str
    duration_ms: float = 0.0
    status: SpanStatus = SpanStatus.OK
    error_type: str | None = None
    tokens: int = 0
    cost_usd: float = 0.0
    metadata: dict = Field(default_factory=dict)


class Trace(BaseModel):
    trace_id: str
    doc_id: str
    spans: list[Span] = Field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.spans)

    @property
    def tokens(self) -> int:
        return sum(s.tokens for s in self.spans)

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    @property
    def ok(self) -> bool:
        return all(s.status == SpanStatus.OK for s in self.spans)
