"""Tests for AnthropicProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm_gateway.types import LLMResponse


class _TestModel(BaseModel):
    answer: str


@pytest.mark.unit
class TestAnthropicProvider:
    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(self) -> None:
        """AnthropicProvider.complete() wraps instructor result in LLMResponse."""
        from llm_gateway.providers.anthropic import AnthropicProvider

        with (
            patch("llm_gateway.providers.anthropic.AsyncAnthropic"),
            patch("llm_gateway.providers.anthropic.instructor") as mock_instructor,
        ):
            provider = AnthropicProvider(api_key="test-key")

            expected = _TestModel(answer="hello")
            # Attach fake _raw_response for token extraction
            raw = MagicMock()
            raw.usage.input_tokens = 100
            raw.usage.output_tokens = 50
            expected._raw_response = raw  # type: ignore[attr-defined]

            mock_instructor.from_anthropic.return_value.messages.create = AsyncMock(
                return_value=expected
            )

            resp = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                response_model=_TestModel,
                model="claude-haiku-4-5-20251001",
            )

            assert isinstance(resp, LLMResponse)
            assert resp.content.answer == "hello"
            assert resp.usage.input_tokens == 100
            assert resp.usage.output_tokens == 50
            assert resp.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_missing_raw_response(self) -> None:
        """Gracefully handles missing _raw_response (usage = 0)."""
        from llm_gateway.providers.anthropic import AnthropicProvider

        with (
            patch("llm_gateway.providers.anthropic.AsyncAnthropic"),
            patch("llm_gateway.providers.anthropic.instructor") as mock_instructor,
        ):
            provider = AnthropicProvider(api_key="test-key")
            expected = _TestModel(answer="ok")
            # No _raw_response attached

            mock_instructor.from_anthropic.return_value.messages.create = AsyncMock(
                return_value=expected
            )

            resp = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                response_model=_TestModel,
                model="claude-haiku-4-5-20251001",
            )

            assert resp.usage.input_tokens == 0
            assert resp.usage.output_tokens == 0

    @pytest.mark.asyncio
    async def test_from_config(self) -> None:
        """from_config factory creates a valid provider."""
        from llm_gateway.config import GatewayConfig
        from llm_gateway.providers.anthropic import AnthropicProvider

        with (
            patch("llm_gateway.providers.anthropic.AsyncAnthropic"),
            patch("llm_gateway.providers.anthropic.instructor"),
        ):
            config = GatewayConfig(
                provider="anthropic",
                api_key="test-key",  # type: ignore[arg-type]
            )
            provider = AnthropicProvider.from_config(config)
            assert isinstance(provider, AnthropicProvider)
