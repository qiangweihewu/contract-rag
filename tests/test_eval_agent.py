from contract_rag.eval.agent_eval import (
    AgentCase,
    AgentRunResult,
    ToolCall,
    evaluate_agent,
    format_agent,
)


def test_all_success_no_takeover():
    def run(case):
        return AgentRunResult(answer="x", citations=["#/b/1"],
                              tool_calls=[ToolCall(name="retrieve", ok=True)], succeeded=True)
    res = evaluate_agent(run, [AgentCase(name="c1", query="q"), AgentCase(name="c2", query="q")])
    assert res["n"] == 2
    assert res["task_success_rate"] == 1.0
    assert res["tool_call_success_rate"] == 1.0
    assert res["hitl_takeover_rate"] == 0.0
    assert res["cited_rate"] == 1.0


def test_mixed_outcomes_are_aggregated():
    def run(case):
        if case.name == "fail":
            return AgentRunResult(answer="", citations=[], hitl_required=True,
                                  tool_calls=[ToolCall(name="retrieve", ok=False)], succeeded=False)
        return AgentRunResult(answer="ok", citations=["#/b/2"],
                              tool_calls=[ToolCall(name="retrieve", ok=True)], succeeded=True)
    res = evaluate_agent(run, [AgentCase(name="ok", query="q"), AgentCase(name="fail", query="q")])
    assert res["task_success_rate"] == 0.5
    assert res["tool_call_success_rate"] == 0.5     # 1 ok of 2 tool-calls
    assert res["hitl_takeover_rate"] == 0.5
    assert res["cited_rate"] == 0.5


def test_empty_cases_are_safe():
    res = evaluate_agent(lambda c: AgentRunResult(), [])
    assert res["n"] == 0
    assert res["task_success_rate"] == 0.0


def test_format_agent_mentions_task_success():
    out = format_agent(evaluate_agent(lambda c: AgentRunResult(succeeded=True),
                                      [AgentCase(name="c", query="q")]))
    assert "task" in out.lower()
