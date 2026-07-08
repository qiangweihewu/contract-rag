"""Human-in-the-loop guard for the agent — mirrors extract/verify.py:
an answer not grounded in a cited block, or below the confidence floor, is
routed to review and never auto-finalized."""
from __future__ import annotations

from pydantic import BaseModel

from contract_rag.agent.models import AgentAnswer, AgentTask
from contract_rag.text import normalize

CONFIDENCE_FLOOR = 0.6


def check_answer(answer: AgentAnswer, threshold: float = CONFIDENCE_FLOOR) -> list[str]:
    reasons: list[str] = []
    if not answer.value:
        reasons.append("empty")
        return reasons
    nv = normalize(answer.value)
    if not nv:
        # value is non-empty but all punctuation/whitespace -> cannot be grounded
        reasons.append("unattributed")
    else:
        attributed = any(nv in normalize(c.text) for c in answer.citations)
        if not attributed:
            reasons.append("unattributed")
    if answer.confidence < threshold:
        reasons.append("low_confidence")
    return reasons


class ReviewItem(BaseModel):
    task: AgentTask
    answer: AgentAnswer
    reasons: list[str]


class HITLQueue:
    def __init__(self) -> None:
        self.items: list[ReviewItem] = []

    def add(self, item: ReviewItem) -> None:
        self.items.append(item)

    def pending(self) -> list[ReviewItem]:
        return list(self.items)
