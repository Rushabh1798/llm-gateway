"""Tests for cost tracking."""

from __future__ import annotations

import pytest

from llm_gateway.cost import (
    CostTracker,
    build_token_usage,
    calculate_cost,
    register_pricing,
)
from llm_gateway.exceptions import CostLimitExceededError
from llm_gateway.types import TokenUsage


@pytest.mark.unit
class TestCalculateCost:
    def test_known_model(self) -> None:
        input_cost, output_cost = calculate_cost("claude-haiku-4-5-20251001", 1_000_000, 1_000_000)
        assert input_cost == pytest.approx(0.80)
        assert output_cost == pytest.approx(4.00)

    def test_unknown_model(self) -> None:
        input_cost, output_cost = calculate_cost("unknown-xyz", 1000, 1000)
        assert input_cost == 0.0
        assert output_cost == 0.0

    def test_register_custom_pricing(self) -> None:
        register_pricing("my-custom-model", 1.0, 5.0)
        input_cost, output_cost = calculate_cost("my-custom-model", 1_000_000, 1_000_000)
        assert input_cost == pytest.approx(1.0)
        assert output_cost == pytest.approx(5.0)


@pytest.mark.unit
class TestBuildTokenUsage:
    def test_builds_with_cost(self) -> None:
        usage = build_token_usage("claude-haiku-4-5-20251001", 500_000, 100_000)
        assert usage.input_tokens == 500_000
        assert usage.output_tokens == 100_000
        assert usage.input_cost_usd == pytest.approx(0.40)
        assert usage.output_cost_usd == pytest.approx(0.40)


@pytest.mark.unit
class TestCostTracker:
    def test_accumulates(self) -> None:
        tracker = CostTracker()
        tracker.record(
            TokenUsage(
                input_tokens=100, output_tokens=50, input_cost_usd=0.01, output_cost_usd=0.02
            )
        )
        tracker.record(
            TokenUsage(
                input_tokens=200, output_tokens=100, input_cost_usd=0.02, output_cost_usd=0.04
            )
        )
        assert tracker.total_tokens == 450
        assert tracker.total_cost_usd == pytest.approx(0.09)
        assert tracker.call_count == 2

    def test_hard_limit(self) -> None:
        tracker = CostTracker(cost_limit_usd=0.05)
        with pytest.raises(CostLimitExceededError):
            tracker.record(TokenUsage(input_cost_usd=0.03, output_cost_usd=0.03))

    def test_warn_threshold(self) -> None:
        tracker = CostTracker(cost_warn_usd=0.01, cost_limit_usd=100.0)
        # Should not raise, just warn
        tracker.record(TokenUsage(input_cost_usd=0.02, output_cost_usd=0.0))
        assert tracker.total_cost_usd == pytest.approx(0.02)

    def test_summary(self) -> None:
        tracker = CostTracker()
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50))
        summary = tracker.summary()
        assert summary["call_count"] == 1
        assert summary["total_tokens"] == 150

    def test_reset(self) -> None:
        tracker = CostTracker()
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50))
        tracker.reset()
        assert tracker.total_tokens == 0
        assert tracker.call_count == 0
