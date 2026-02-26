"""Integration tests for FakeLLMProvider shipped with llm-gateway.

Validates that FakeLLMProvider works through the full LLMClient stack:
config-driven creation, cost tracking, context manager, response_factory.

Run with: pytest -m dry_run -v
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import BaseModel, Field

from llm_gateway import (
    FakeCall,
    FakeLLMProvider,
    GatewayConfig,
    LLMClient,
    build_provider,
)
from llm_gateway.types import LLMMessage

# ── Test models ──────────────────────────────────────────────────


class Answer(BaseModel):
    """Simple response model for tests."""

    text: str = Field(description="The answer text.")


class Score(BaseModel):
    """Numeric response model for multi-model tests."""

    value: int = Field(description="A numeric score.")


# ── Tests ────────────────────────────────────────────────────────


@pytest.mark.dry_run
class TestFakeProviderIntegration:
    """Verify FakeLLMProvider works through the full LLMClient stack."""

    async def test_client_with_injected_fake(self) -> None:
        """LLMClient(provider_instance=fake) returns correct response."""
        fake = FakeLLMProvider()
        fake.set_response(Answer, Answer(text="42"))

        config = GatewayConfig(
            provider="fake",
            model="test-model",
            trace_enabled=False,
            log_format="console",
        )
        client = LLMClient(config=config, provider_instance=fake)

        resp = await client.complete(
            messages=[{"role": "user", "content": "What is 6*7?"}],
            response_model=Answer,
        )

        assert resp.content.text == "42"
        assert resp.provider == "fake"
        await client.close()

    async def test_client_cost_tracking_with_fake(self) -> None:
        """Client cost_summary() reflects fake provider token usage."""
        fake = FakeLLMProvider(default_input_tokens=200, default_output_tokens=100)
        fake.set_response(Answer, Answer(text="test"))

        config = GatewayConfig(
            provider="fake",
            model="test-model",
            trace_enabled=False,
            log_format="console",
        )
        client = LLMClient(config=config, provider_instance=fake)

        await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            response_model=Answer,
        )

        summary = client.cost_summary()
        assert summary["call_count"] == 1
        assert summary["total_input_tokens"] == 200
        assert summary["total_output_tokens"] == 100
        assert summary["total_tokens"] == 300
        await client.close()

    async def test_client_context_manager_with_fake(self) -> None:
        """async with LLMClient(provider_instance=fake) works."""
        fake = FakeLLMProvider()
        fake.set_response(Answer, Answer(text="ctx"))

        config = GatewayConfig(
            provider="fake",
            model="test-model",
            trace_enabled=False,
            log_format="console",
        )

        async with LLMClient(config=config, provider_instance=fake) as client:
            resp = await client.complete(
                messages=[{"role": "user", "content": "hi"}],
                response_model=Answer,
            )
            assert resp.content.text == "ctx"

    async def test_fake_via_config_provider_name(self) -> None:
        """GatewayConfig(provider='fake') auto-creates FakeLLMProvider via registry."""
        config = GatewayConfig(
            provider="fake",
            model="test-model",
            trace_enabled=False,
            log_format="console",
        )
        provider = build_provider(config)
        assert isinstance(provider, FakeLLMProvider)

    async def test_response_factory_with_client(self) -> None:
        """response_factory hook works through full LLMClient.complete() path."""

        def factory(
            model_cls: type[Answer],
            messages: Sequence[LLMMessage],
        ) -> Answer:
            return Answer(text="dynamic")

        fake = FakeLLMProvider(response_factory=factory)
        config = GatewayConfig(
            provider="fake",
            model="test-model",
            trace_enabled=False,
            log_format="console",
        )

        async with LLMClient(config=config, provider_instance=fake) as client:
            resp = await client.complete(
                messages=[{"role": "user", "content": "hi"}],
                response_model=Answer,
            )
            assert resp.content.text == "dynamic"

    async def test_multiple_calls_accumulate_cost(self) -> None:
        """Multiple complete() calls accumulate in client.total_cost_usd."""
        fake = FakeLLMProvider()
        fake.set_response(Answer, Answer(text="a"))
        fake.set_response(Score, Score(value=10))

        config = GatewayConfig(
            provider="fake",
            model="test-model",
            trace_enabled=False,
            log_format="console",
        )

        async with LLMClient(config=config, provider_instance=fake) as client:
            await client.complete(
                messages=[{"role": "user", "content": "one"}],
                response_model=Answer,
            )
            await client.complete(
                messages=[{"role": "user", "content": "two"}],
                response_model=Score,
            )

            assert client.call_count == 2
            assert client.total_tokens == 300  # 2 * (100 + 50)
            assert fake.call_count == 2

    async def test_fake_call_dataclass_exported(self) -> None:
        """FakeCall is importable from llm_gateway top-level."""
        fake = FakeLLMProvider()
        fake.set_response(Answer, Answer(text="exported"))

        await fake.complete(
            messages=[{"role": "user", "content": "hi"}],
            response_model=Answer,
            model="m",
        )

        call = fake.calls[0]
        assert isinstance(call, FakeCall)
        assert call.response_model is Answer
