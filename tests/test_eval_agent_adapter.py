from __future__ import annotations

from contract_rag.agent.models import (
    AgentAnswer, AgentResult, AgentState, AgentStatus, AgentTask, Citation, Step,
)
from contract_rag.eval.agent_eval import AgentRunResult, to_run_result


def _result(status: AgentStatus, *, value: str, conf: float, cites, steps) -> AgentResult:
    state = AgentState(
        task=AgentTask(question="q"),
        status=status,
        steps=steps,
        answer=AgentAnswer(value=value, confidence=conf, citations=cites),
    )
    return AgentResult(state=state, trace_id="t1")


def test_done_result_maps_to_succeeded_with_citations():
    res = _result(
        AgentStatus.DONE, value="New York", conf=0.7,
        cites=[Citation(block_id="b3", text="governed by New York")],
        steps=[Step(tool="retrieve", input={}, output={}, ok=True),
               Step(tool="extract_field", input={}, output={}, ok=True)],
    )
    run = to_run_result(res)
    assert isinstance(run, AgentRunResult)
    assert run.answer == "New York"
    assert run.citations == ["b3"]
    assert [tc.name for tc in run.tool_calls] == ["retrieve", "extract_field"]
    assert all(tc.ok for tc in run.tool_calls)
    assert run.succeeded is True
    assert run.hitl_required is False


def test_needs_hitl_maps_to_not_succeeded():
    res = _result(
        AgentStatus.NEEDS_HITL, value="", conf=0.0, cites=[],
        steps=[Step(tool="retrieve", input={}, output={}, ok=True)],
    )
    run = to_run_result(res)
    assert run.succeeded is False
    assert run.hitl_required is True
    assert run.citations == []


def test_none_answer_is_safe():
    state = AgentState(task=AgentTask(question="q"), status=AgentStatus.FAILED, answer=None)
    run = to_run_result(AgentResult(state=state, trace_id=None))
    assert run.answer == ""
    assert run.succeeded is False
    assert run.hitl_required is False
