from __future__ import annotations

# USD per 1K tokens (blended input+output, rough). Credential-free backends
# (rule, hashing) are intentionally absent → cost_for returns 0.0 for them.
PRICES: dict[str, float] = {
    "gpt-4o": 0.005,
}


def cost_for(model: str, tokens: int) -> float:
    return PRICES.get(model, 0.0) * tokens / 1000.0


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token). Used when a backend
    doesn't return real usage (injected fakes, servers without a usage field)."""
    return len(text) // 4
