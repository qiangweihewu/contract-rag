"""Agent runner: drive the state machine, retry/fallback tool calls, record an
obs span per attempt, and route un-grounded answers to HITL (never auto-finalize)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from contract_rag.agent.hitl import HITLQueue, ReviewItem, check_answer

if TYPE_CHECKING:
    from contract_rag.verticals.base import Vertical
from contract_rag.agent.models import (
    AgentAnswer, AgentResult, AgentState, AgentStatus, AgentTask, Citation, Step,
)


def run_with_retry(tool, inp: dict, tracer, trace, *, retries: int = 2):
    last_error: str | None = None
    for _ in range(retries + 1):
        try:
            with tracer.span(trace, tool.name):
                out = tool.run(inp)
            return out, None
        except Exception as exc:  # span already recorded the error; retry
            last_error = str(exc) or type(exc).__name__
    return None, last_error


def finalize(state: AgentState) -> AgentAnswer:
    """Builds the answer from the extract_field + cite steps — coupled to the RulePlanner retrieve→extract→cite shape; an LLM path that answers without an extract_field step yields an empty answer (→ NEEDS_HITL), never a wrong DONE."""
    ext = next((s for s in state.steps if s.tool == "extract_field" and s.ok), None)
    if ext is None:
        return AgentAnswer()
    citations = [
        Citation(block_id=s.output["block_id"], text=s.output["text"])
        for s in state.steps
        if s.tool == "cite" and s.ok
    ]
    return AgentAnswer(
        value=ext.output.get("value", ""),
        confidence=ext.output.get("confidence", 0.0),
        citations=citations,
    )


def run_agent(task: AgentTask, planner, tools, *, tracer=None, hitl=None, max_steps: int = 10) -> AgentResult:
    from contract_rag.obs.tracer import NoopTracer

    tracer = tracer or NoopTracer()
    hitl = hitl or HITLQueue()
    state = AgentState(task=task)
    trace = tracer.start(doc_id=task.doc_id)

    while state.status == AgentStatus.RUNNING and len(state.steps) < max_steps:
        call = planner.next_action(state)
        if call is None:
            break
        tool = tools.get(call.tool)
        if tool is None:
            state.steps.append(Step(tool=call.tool, input=call.input, output={},
                                    ok=False, error="unknown_tool"))
            state.status = AgentStatus.FAILED
            break
        out, err = run_with_retry(tool, call.input, tracer, trace)
        if err is not None:
            state.steps.append(Step(tool=call.tool, input=call.input, output={},
                                    ok=False, error=err))
            state.status = AgentStatus.FAILED
            break
        state.steps.append(Step(tool=call.tool, input=call.input, output=out, ok=True))

    if state.status == AgentStatus.RUNNING:
        answer = finalize(state)
        state.answer = answer
        reasons = check_answer(answer)
        if reasons:
            state.status = AgentStatus.NEEDS_HITL
            hitl.add(ReviewItem(task=task, answer=answer, reasons=reasons))
        else:
            state.status = AgentStatus.DONE

    tracer.finish(trace)
    return AgentResult(state=state, trace_id=trace.trace_id)


def build_agent_tools(ir, index, extractor, principal=None, vertical: Vertical | None = None) -> dict:
    """Wire the four tools into a name->tool registry for run_agent."""
    from contract_rag.agent.tools import (
        CheckClauseTool, CiteTool, ExtractFieldTool, RetrieveTool,
    )

    tools = [
        RetrieveTool(index, principal=principal),
        ExtractFieldTool(extractor, ir),
        CheckClauseTool(ir, vertical=vertical),
        CiteTool(ir),
    ]
    return {t.name: t for t in tools}
