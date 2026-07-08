from contract_rag.obs.pricing import PRICES, cost_for


def test_cost_for_known_model_scales_with_tokens():
    PRICES["test-model"] = 2.0  # $2 / 1K tokens
    assert cost_for("test-model", 0) == 0.0
    assert cost_for("test-model", 500) == 1.0
    assert cost_for("test-model", 1000) == 2.0


def test_cost_for_unknown_model_is_free():
    assert cost_for("rule", 12345) == 0.0
    assert cost_for("hashing", 999) == 0.0
