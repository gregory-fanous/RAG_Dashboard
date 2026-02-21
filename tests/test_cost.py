import pytest

from rag_eval.cost import token_cost_usd


def test_token_cost_usd_calculation():
    value = token_cost_usd(
        prompt_tokens=2000,
        completion_tokens=1000,
        prompt_price_per_1k=0.003,
        completion_price_per_1k=0.015,
    )
    assert value == pytest.approx(0.021)


def test_token_cost_usd_rejects_negative_values():
    with pytest.raises(ValueError):
        token_cost_usd(-1, 10, 0.003, 0.015)
