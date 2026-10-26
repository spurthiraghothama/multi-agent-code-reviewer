"""Token pricing and cost calculation.

Prices are USD per million tokens (Anthropic first-party API). Cache writes (5-minute TTL) cost
1.25x the input price; cache reads are listed per model because the discount differs by model.
Thinking tokens are billed as output tokens and are already included in usage.output_tokens.

Update this table when prices change - it is the single source of truth for the cost metrics in
the benchmark report.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float

    @property
    def cache_write(self) -> float:
        return self.input * 1.25


PRICES: dict[str, Price] = {
    "claude-fable-5-1": Price(10.00, 50.00, 0.25),
    "claude-opus-5-5": Price(4.00, 20.00, 0.20),
    "claude-opus-5": Price(5.00, 25.00, 0.50),
    "claude-opus-4-8": Price(5.00, 25.00, 0.50),
    "claude-sonnet-5-5": Price(2.00, 10.00, 0.20),
    "claude-sonnet-5": Price(2.00, 10.00, 0.20),
    "claude-haiku-5-5": Price(0.10, 0.50, 0.01),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_write_tokens: int = 0,
             cache_read_tokens: int = 0) -> float:
    p = PRICES.get(model)
    if p is None:
        return 0.0  # unknown model: report tokens, not a made-up price
    return (input_tokens * p.input + output_tokens * p.output + cache_write_tokens * p.cache_write
            + cache_read_tokens * p.cache_read) / 1_000_000
