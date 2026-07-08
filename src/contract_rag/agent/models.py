from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    NEEDS_HITL = "needs_hitl"
    FAILED = "failed"


class AgentTask(BaseModel):
    question: str
    field: str | None = None
    doc_id: str = "doc"


class Citation(BaseModel):
    block_id: str
    text: str


class AgentAnswer(BaseModel):
    value: str = ""
    confidence: float = 0.0
    citations: list[Citation] = Field(default_factory=list)


class ToolCall(BaseModel):
    tool: str
    input: dict = Field(default_factory=dict)


class Step(BaseModel):
    tool: str
    input: dict
    output: dict
    ok: bool = True
    error: str | None = None


class AgentState(BaseModel):
    task: AgentTask
    status: AgentStatus = AgentStatus.RUNNING
    steps: list[Step] = Field(default_factory=list)
    answer: AgentAnswer | None = None


class AgentResult(BaseModel):
    state: AgentState
    trace_id: str | None = None
