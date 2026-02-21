from __future__ import annotations


def token_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    prompt_price_per_1k: float,
    completion_price_per_1k: float,
) -> float:
    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError("Token counts must be non-negative")
    prompt_cost = (prompt_tokens / 1000.0) * prompt_price_per_1k
    completion_cost = (completion_tokens / 1000.0) * completion_price_per_1k
    return prompt_cost + completion_cost
