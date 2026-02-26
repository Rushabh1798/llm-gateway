"""Token pricing registry and cost tracking."""

from __future__ import annotations

import logging
from typing import Any

from llm_gateway.exceptions import CostLimitExceededError
from llm_gateway.types import TokenUsage

logger = logging.getLogger(__name__)

# ── Pricing Registry (USD per 1 million tokens) ────────────────
_PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-5-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    # OpenAI (examples — update as needed)
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}


def register_pricing(model: str, input_per_1m: float, output_per_1m: float) -> None:
    """Register or update pricing for a model.

    Args:
        model: Model identifier string.
        input_per_1m: Cost in USD per 1M input tokens.
        output_per_1m: Cost in USD per 1M output tokens.
    """
    _PRICING[model] = {"input": input_per_1m, "output": output_per_1m}


def get_pricing(model: str) -> dict[str, float] | None:
    """Return pricing dict for a model, or None if unknown."""
    return _PRICING.get(model)


def calculate_cost(
    model: str, input_tokens: int, output_tokens: int
) -> tuple[float, float]:
    """Calculate USD cost for a given token count.

    Returns:
        Tuple of (input_cost_usd, output_cost_usd). Both 0.0 if model unknown.
    """
    pricing = _PRICING.get(model)
    if pricing is None:
        return 0.0, 0.0
    input_cost = input_tokens * pricing["input"] / 1_000_000
    output_cost = output_tokens * pricing["output"] / 1_000_000
    return input_cost, output_cost


def build_token_usage(
    model: str, input_tokens: int, output_tokens: int
) -> TokenUsage:
    """Build a TokenUsage with cost calculated from the pricing registry."""
    input_cost, output_cost = calculate_cost(model, input_tokens, output_tokens)
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
    )


class CostTracker:
    """Accumulates token usage and cost across multiple LLM calls.

    Supports cost guardrails (warn and hard limit).
    """

    def __init__(
        self,
        cost_limit_usd: float | None = None,
        cost_warn_usd: float | None = None,
    ) -> None:
        self._cost_limit = cost_limit_usd
        self._cost_warn = cost_warn_usd
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._call_count: int = 0
        self._warned: bool = False

    def record(self, usage: TokenUsage) -> None:
        """Record a single LLM call's usage and check guardrails."""
        self._total_input_tokens += usage.input_tokens
        self._total_output_tokens += usage.output_tokens
        self._total_cost_usd += usage.total_cost_usd
        self._call_count += 1

        self._check_guardrails()

    def _check_guardrails(self) -> None:
        """Enforce cost warning and hard limit."""
        if (
            self._cost_warn
            and not self._warned
            and self._total_cost_usd >= self._cost_warn
        ):
            self._warned = True
            logger.warning(
                "LLM cost warning threshold reached: $%.4f >= $%.4f",
                self._total_cost_usd,
                self._cost_warn,
            )

        if self._cost_limit and self._total_cost_usd >= self._cost_limit:
            raise CostLimitExceededError(self._total_cost_usd, self._cost_limit)

    @property
    def total_cost_usd(self) -> float:
        """Cumulative cost in USD."""
        return self._total_cost_usd

    @property
    def total_tokens(self) -> int:
        """Cumulative total tokens."""
        return self._total_input_tokens + self._total_output_tokens

    @property
    def call_count(self) -> int:
        """Number of LLM calls recorded."""
        return self._call_count

    def summary(self) -> dict[str, Any]:
        """Return a summary dict suitable for logging or span attributes."""
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self._total_cost_usd, 6),
            "call_count": self._call_count,
        }

    def reset(self) -> None:
        """Reset all accumulators."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost_usd = 0.0
        self._call_count = 0
        self._warned = False
