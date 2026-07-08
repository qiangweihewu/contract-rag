from contract_rag.agent.hitl import HITLQueue, ReviewItem, check_answer
from contract_rag.agent.models import AgentAnswer, AgentTask, Citation


def test_check_answer_clean_when_attributed_and_confident():
    ans = AgentAnswer(value="New York", confidence=0.9,
                      citations=[Citation(block_id="b1", text="governed by the State of New York")])
    assert check_answer(ans) == []


def test_check_answer_flags_unattributed_and_low_confidence_and_empty():
    assert "empty" in check_answer(AgentAnswer(value="", confidence=0.9))
    unattributed = AgentAnswer(value="California", confidence=0.9,
                               citations=[Citation(block_id="b1", text="New York")])
    assert "unattributed" in check_answer(unattributed)
    low = AgentAnswer(value="New York", confidence=0.3,
                      citations=[Citation(block_id="b1", text="New York")])
    assert "low_confidence" in check_answer(low)


def test_hitl_queue_accumulates():
    q = HITLQueue()
    q.add(ReviewItem(task=AgentTask(question="q"), answer=AgentAnswer(value=""), reasons=["empty"]))
    assert len(q.pending()) == 1
    assert q.pending()[0].reasons == ["empty"]


def test_check_answer_flags_value_that_normalizes_to_empty():
    # non-empty but all-punctuation value normalizes to "" -> must NOT count as grounded
    ans = AgentAnswer(value=".", confidence=0.9,
                      citations=[Citation(block_id="b1", text="anything at all")])
    assert "unattributed" in check_answer(ans)
