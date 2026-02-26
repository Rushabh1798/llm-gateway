"""Integration tests that call real providers.

Run with: pytest -m integration
Requires actual API keys in environment.
"""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel

from llm_gateway import GatewayConfig, LLMClient


class _SimpleAnswer(BaseModel):
    greeting: str


@pytest.mark.integration
class TestLiveAnthropic:
    @pytest.mark.asyncio
    async def test_anthropic_round_trip(self) -> None:
        """Real Anthropic API call with cost tracking."""
        if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("LLM_API_KEY"):
            pytest.skip("No API key available")

        config = GatewayConfig(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            cost_limit_usd=1.0,
        )
        async with LLMClient(config=config) as client:
            resp = await client.complete(
                messages=[{"role": "user", "content": "Say hello in one word."}],
                response_model=_SimpleAnswer,
            )
            assert resp.content.greeting
            assert resp.usage.input_tokens > 0
            assert resp.usage.total_cost_usd > 0
            assert client.total_cost_usd > 0


@pytest.mark.integration
class TestLiveLocalClaude:
    @pytest.mark.asyncio
    async def test_local_claude_round_trip(self) -> None:
        """Real local claude CLI call."""
        import shutil

        if not shutil.which("claude"):
            pytest.skip("claude CLI not in PATH")

        config = GatewayConfig(provider="local_claude")
        async with LLMClient(config=config) as client:
            resp = await client.complete(
                messages=[{"role": "user", "content": "Say hello in one word."}],
                response_model=_SimpleAnswer,
            )
            assert resp.content.greeting
            assert resp.provider == "local_claude"
