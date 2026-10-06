from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UsageTracker:
    """Private in-process counters. Not exposed through any endpoint; the
    per-request numbers are also written to the logs."""

    price_in_per_million: float = 0.0
    price_out_per_million: float = 0.0
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.price_in_per_million + output_tokens * self.price_out_per_million
        ) / 1_000_000

    def record(self, input_tokens: int, output_tokens: int) -> float:
        self.requests += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        return self.estimate_cost(input_tokens, output_tokens)

    def snapshot(self) -> dict:
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost": self.estimate_cost(self.input_tokens, self.output_tokens),
        }
