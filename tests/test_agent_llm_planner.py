from contract_rag.agent.models import AgentState, AgentTask
from contract_rag.agent.planner import LLMPlanner


class _FakeClient:
    def __init__(self, content):
        msg = type("M", (), {"content": content})()
        choice = type("C", (), {"message": msg})()
        resp = type("R", (), {"choices": [choice]})()
        create = lambda **kw: resp
        completions = type("Comp", (), {"create": staticmethod(create)})()
        self.chat = type("Chat", (), {"completions": completions})()


def test_llm_planner_parses_tool_call_from_model():
    client = _FakeClient('{"tool": "retrieve", "input": {"query": "governing law", "k": 5}}')
    planner = LLMPlanner(client, model="x", allowed_tools=["retrieve", "extract_field", "cite"])
    action = planner.next_action(AgentState(task=AgentTask(question="What is the governing law?")))
    assert action.tool == "retrieve"
    assert action.input["query"] == "governing law"


def test_llm_planner_returns_none_on_finish():
    client = _FakeClient('{"tool": "finish", "input": {}}')
    planner = LLMPlanner(client, model="x", allowed_tools=["retrieve"])
    assert planner.next_action(AgentState(task=AgentTask(question="q"))) is None
