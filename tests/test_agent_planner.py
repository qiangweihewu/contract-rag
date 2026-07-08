from contract_rag.agent.models import AgentState, AgentTask, Step
from contract_rag.agent.planner import RulePlanner, infer_field


def test_infer_field_keyword_routing():
    assert infer_field("What is the governing law?") == "governing_law"
    assert infer_field("When is the effective date?") == "effective_date"
    assert infer_field("Tell me a joke") is None


def test_rule_planner_sequences_retrieve_extract_cite_then_finish():
    planner = RulePlanner()
    state = AgentState(task=AgentTask(question="What is the governing law?"))

    a1 = planner.next_action(state)
    assert a1.tool == "retrieve" and a1.input["query"] == "What is the governing law?"
    state.steps.append(Step(tool="retrieve", input=a1.input, output={"chunks": []}))

    a2 = planner.next_action(state)
    assert a2.tool == "extract_field" and a2.input["field"] == "governing_law"
    state.steps.append(Step(tool="extract_field", input=a2.input,
                            output={"value": "New York", "source_block_id": "b1", "confidence": 0.9}))

    a3 = planner.next_action(state)
    assert a3.tool == "cite" and a3.input["block_id"] == "b1"
    state.steps.append(Step(tool="cite", input=a3.input, output={"block_id": "b1", "text": "..."}))

    assert planner.next_action(state) is None  # finished


def test_rule_planner_finishes_after_extract_when_no_source_block():
    planner = RulePlanner()
    state = AgentState(task=AgentTask(question="What is the governing law?", field="governing_law"))
    state.steps.append(Step(tool="retrieve", input={}, output={"chunks": []}))
    state.steps.append(Step(tool="extract_field", input={"field": "governing_law"},
                            output={"value": "", "source_block_id": None, "confidence": 0.0}))
    assert planner.next_action(state) is None  # nothing to cite -> finish
