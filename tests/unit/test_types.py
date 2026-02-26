"""Tests for core types."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from llm_gateway.types import LLMResponse, TokenUsage


@pytest.mark.unit
class TestTokenUsage:
    def test_total_tokens(self) -> None:
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_total_cost(self) -> None:
        usage = TokenUsage(input_cost_usd=0.01, output_cost_usd=0.02)
        assert usage.total_cost_usd == pytest.approx(0.03)

    def test_defaults(self) -> None:
        usage = TokenUsage()
        assert usage.total_tokens == 0
        assert usage.total_cost_usd == 0.0

    def test_frozen(self) -> None:
        usage = TokenUsage(input_tokens=10)
        with pytest.raises(AttributeError):
            usage.input_tokens = 20  # type: ignore[misc]


@pytest.mark.unit
class TestLLMResponse:
    def test_generic_content(self) -> None:
        class Answer(BaseModel):
            text: str

        resp = LLMResponse(
            content=Answer(text="hello"),
            usage=TokenUsage(),
            model="test-model",
            provider="test",
        )
        assert resp.content.text == "hello"
        assert resp.provider == "test"
