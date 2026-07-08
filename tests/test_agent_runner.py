from contract_rag.agent.hitl import HITLQueue
from contract_rag.agent.models import AgentStatus, AgentTask, ToolCall
from contract_rag.agent.runner import run_agent
from contract_rag.obs.store import InMemoryTraceStore
from contract_rag.obs.tracer import Tracer


class _ScriptedPlanner:
    """Emits a fixed sequence of ToolCalls, then None."""
    def __init__(self, calls):
        self._calls = calls

    def next_action(self, state):
        i = len(state.steps)
        return self._calls[i] if i < len(self._calls) else None


class _OkTool:
    def __init__(self, name, out):
        self.name = name
        self._out = out

    def run(self, inp):
        return self._out


class _FlakyTool:
    """Fails the first `fails` calls, then succeeds."""
    def __init__(self, name, out, fails):
        self.name = name
        self._out = out
        self._fails = fails
        self.calls = 0

    def run(self, inp):
        self.calls += 1
        if self.calls <= self._fails:
            raise RuntimeError("boom")
        return self._out


def _tools(*tools):
    return {t.name: t for t in tools}


def test_run_agent_happy_path_done_with_answer_and_trace():
    planner = _ScriptedPlanner([
        ToolCall(tool="retrieve", input={"query": "q"}),
        ToolCall(tool="extract_field", input={"field": "governing_law"}),
        ToolCall(tool="cite", input={"block_id": "b1"}),
    ])
    tools = _tools(
        _OkTool("retrieve", {"chunks": []}),
        _OkTool("extract_field", {"value": "New York", "source_block_id": "b1", "confidence": 0.9}),
        _OkTool("cite", {"block_id": "b1", "text": "governed by the State of New York"}),
    )
    store = InMemoryTraceStore()
    result = run_agent(AgentTask(question="q"), planner, tools, tracer=Tracer(store=store))

    assert result.state.status == AgentStatus.DONE
    assert result.state.answer.value == "New York"
    assert result.state.answer.citations[0].block_id == "b1"
    # one trace persisted, with a span per tool step
    assert [s.name for s in store.all()[0].spans] == ["retrieve", "extract_field", "cite"]


def test_run_agent_routes_ungrounded_answer_to_hitl():
    planner = _ScriptedPlanner([
        ToolCall(tool="retrieve", input={"query": "q"}),
        ToolCall(tool="extract_field", input={"field": "governing_law"}),
    ])
    tools = _tools(
        _OkTool("retrieve", {"chunks": []}),
        # value not present in any citation -> unattributed
        _OkTool("extract_field", {"value": "California", "source_block_id": None, "confidence": 0.9}),
    )
    hitl = HITLQueue()
    result = run_agent(AgentTask(question="q"), planner, tools, hitl=hitl)
    assert result.state.status == AgentStatus.NEEDS_HITL
    assert len(hitl.pending()) == 1
    assert "unattributed" in hitl.pending()[0].reasons


def test_run_agent_retries_flaky_tool_then_succeeds():
    flaky = _FlakyTool("retrieve", {"chunks": []}, fails=1)
    planner = _ScriptedPlanner([ToolCall(tool="retrieve", input={"query": "q"})])
    store = InMemoryTraceStore()
    result = run_agent(AgentTask(question="q"), planner, _tools(flaky), tracer=Tracer(store=store))
    assert flaky.calls == 2  # one failure + one success
    # two spans for retrieve (error attempt + ok attempt); status is the error from attempt 1
    span_names = [s.name for s in store.all()[0].spans]
    assert span_names.count("retrieve") == 2
    assert result.state.status != AgentStatus.FAILED


def test_run_agent_fails_on_unknown_tool():
    planner = _ScriptedPlanner([ToolCall(tool="missing", input={})])
    result = run_agent(AgentTask(question="q"), planner, {})
    assert result.state.status == AgentStatus.FAILED
    assert result.state.steps[0].error == "unknown_tool"
