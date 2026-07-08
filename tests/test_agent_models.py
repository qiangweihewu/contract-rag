from contract_rag.agent.models import (
    AgentAnswer, AgentResult, AgentState, AgentStatus, AgentTask, Citation, Step, ToolCall,
)


def test_agent_state_defaults_to_running_with_no_steps():
    state = AgentState(task=AgentTask(question="What is the governing law?"))
    assert state.status == AgentStatus.RUNNING
    assert state.steps == []
    assert state.answer is None


def test_step_and_answer_round_trip():
    step = Step(tool="retrieve", input={"query": "q"}, output={"chunks": []})
    assert step.ok is True and step.error is None
    ans = AgentAnswer(value="New York", confidence=0.9,
                      citations=[Citation(block_id="b1", text="...New York...")])
    result = AgentResult(state=AgentState(task=AgentTask(question="q"), answer=ans), trace_id="t1")
    assert result.state.answer.value == "New York"
    assert result.trace_id == "t1"
    assert ToolCall(tool="cite", input={"block_id": "b1"}).tool == "cite"
