"""Planner: decide the agent's next tool call. RulePlanner is deterministic and
credential-free (the CI/default policy); LLMPlanner (gated) is added later."""
from __future__ import annotations

import json
import re
from typing import Protocol

from contract_rag.agent.models import AgentState, ToolCall

# Question keyword -> ContractFacts field. First match wins.
_FIELD_KEYWORDS: list[tuple[str, str]] = [
    ("governing_law", r"governing law|jurisdiction|governed by"),
    ("effective_date", r"effective date|\bdated\b|commenc"),
    ("counterparty", r"counterpart|parties|\bbetween\b"),
    ("termination_notice_days", r"terminat|notice period"),
    ("auto_renewal", r"renew|auto-?renew"),
    ("total_value", r"total value|\bamount\b|\bprice\b|\bfee\b"),
]


def infer_field(question: str) -> str | None:
    q = question.lower()
    for field, pattern in _FIELD_KEYWORDS:
        if re.search(pattern, q):
            return field
    return None


class Planner(Protocol):
    def next_action(self, state: AgentState) -> ToolCall | None: ...


class RulePlanner:
    def next_action(self, state: AgentState) -> ToolCall | None:
        done = [s.tool for s in state.steps]
        task = state.task
        if "retrieve" not in done:
            return ToolCall(tool="retrieve", input={"query": task.question, "k": 5})
        field = task.field or infer_field(task.question)
        if field and "extract_field" not in done:
            return ToolCall(tool="extract_field", input={"field": field})
        if "extract_field" in done and "cite" not in done:
            ext = next((s for s in state.steps if s.tool == "extract_field"), None)
            block_id = ext.output.get("source_block_id") if ext else None
            if block_id:
                return ToolCall(tool="cite", input={"block_id": block_id})
        return None


class LLMPlanner:
    """Gated planner: an LLM picks the next tool. Construct only after
    config.assert_backend_allowed — the client is injected so tests need no network."""

    _SYSTEM = (
        "You orchestrate a contract-analysis agent. Given the task and steps so far, "
        "reply with ONLY a JSON object {\"tool\": <name>, \"input\": {...}}. "
        "Use tool \"finish\" when the answer is ready."
    )

    def __init__(self, client, model: str, allowed_tools: list[str]) -> None:
        self.client = client
        self.model = model
        self.allowed_tools = allowed_tools

    def next_action(self, state: AgentState) -> ToolCall | None:
        prompt = {
            "question": state.task.question,
            "field": state.task.field,
            "allowed_tools": self.allowed_tools,
            "steps": [{"tool": s.tool, "output": s.output} for s in state.steps],
        }
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._SYSTEM},
                {"role": "user", "content": json.dumps(prompt)},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        if data.get("tool") in (None, "finish"):
            return None
        return ToolCall(tool=data["tool"], input=data.get("input", {}))
