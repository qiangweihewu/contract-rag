"""Agent eval (spec §86, gate G1): tool-call success, HITL-takeover rate, end-to-end
task success — the agent eval dimension the rubric demands (§3.2/§3.4).

`AgentRunResult` is the MEASUREMENT-SIDE CONTRACT: the per-task output shape this eval
needs from Alex's S3 agent. Published here so Lane C is unblocked before S3 lands; at
integration it is reconciled with the agent's own published I/O stub (one adapter, if
the field names differ). Credential-free: the eval drives the agent via an injected
`run_agent` callable, so tests use a fake agent and never touch a model."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from contract_rag.agent.models import AgentResult


class ToolCall(BaseModel):
    name: str
    ok: bool


class AgentRunResult(BaseModel):
    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    hitl_required: bool = False
    succeeded: bool = False


class AgentCase(BaseModel):
    name: str
    query: str
    expect_citation: bool = True


def evaluate_agent(
    run_agent: Callable[[AgentCase], AgentRunResult], cases: list[AgentCase]
) -> dict:
    n = len(cases)
    if n == 0:
        return {"n": 0, "task_success_rate": 0.0, "tool_call_success_rate": 0.0,
                "hitl_takeover_rate": 0.0, "cited_rate": 0.0}
    results = [(c, run_agent(c)) for c in cases]
    tool_calls = [tc for _, r in results for tc in r.tool_calls]
    cite_expected = [(c, r) for c, r in results if c.expect_citation]
    return {
        "n": n,
        "task_success_rate": sum(1 for _, r in results if r.succeeded) / n,
        "tool_call_success_rate": (
            sum(1 for tc in tool_calls if tc.ok) / len(tool_calls) if tool_calls else 0.0
        ),
        "hitl_takeover_rate": sum(1 for _, r in results if r.hitl_required) / n,
        "cited_rate": (
            sum(1 for _, r in cite_expected if r.citations) / len(cite_expected)
            if cite_expected else 0.0
        ),
    }


def to_run_result(result: "AgentResult", *, expect_citation: bool = True) -> AgentRunResult:
    """Adapt the agent's own AgentResult into the measurement-side AgentRunResult so
    evaluate_agent can score the real run_agent. `expect_citation` is accepted for
    symmetry with AgentCase; mapping itself does not depend on it."""
    from contract_rag.agent.models import AgentStatus

    state = result.state
    ans = state.answer
    return AgentRunResult(
        answer=ans.value if ans else "",
        citations=[c.block_id for c in ans.citations] if ans else [],
        tool_calls=[ToolCall(name=s.tool, ok=s.ok) for s in state.steps],
        hitl_required=state.status == AgentStatus.NEEDS_HITL,
        succeeded=state.status == AgentStatus.DONE,
    )


def format_agent(res: dict) -> str:
    return "\n".join([
        "=== agent eval ===",
        f"cases:                  {res['n']}",
        f"task_success_rate:      {res['task_success_rate']:.3f}",
        f"tool_call_success_rate: {res['tool_call_success_rate']:.3f}",
        f"hitl_takeover_rate:     {res['hitl_takeover_rate']:.3f}",
        f"cited_rate:             {res['cited_rate']:.3f}",
    ])
