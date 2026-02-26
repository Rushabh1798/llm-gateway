"""Tests for observability tracing."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import llm_gateway.observability.tracing as tracing_mod
from llm_gateway.observability.tracing import (
    configure_tracing,
    disable_tracing,
    get_tracer,
    traced_llm_call,
)
from llm_gateway.types import LLMResponse, TokenUsage


@pytest.mark.unit
class TestConfigureTracing:
    """Tests for configure_tracing() function."""

    def setup_method(self) -> None:
        disable_tracing()

    def teardown_method(self) -> None:
        disable_tracing()

    def test_none_exporter_disables_tracer(self) -> None:
        configure_tracing(exporter="none")
        assert get_tracer() is None

    def test_default_exporter_is_none(self) -> None:
        configure_tracing()
        assert get_tracer() is None

    def test_disable_tracing_clears_tracer(self) -> None:
        disable_tracing()
        assert get_tracer() is None

    def test_get_tracer_returns_none_by_default(self) -> None:
        assert get_tracer() is None

    def test_console_exporter_creates_tracer(self) -> None:
        """Console exporter sets up a real tracer when OTEL is installed."""
        if not tracing_mod.HAS_OTEL:
            pytest.skip("opentelemetry not installed")
        configure_tracing(exporter="console", service_name="test-svc")
        assert get_tracer() is not None

    def test_otlp_without_otlp_package_falls_back(self) -> None:
        """OTLP exporter gracefully falls back when exporter package is missing."""
        if not tracing_mod.HAS_OTEL:
            pytest.skip("opentelemetry not installed")
        original = tracing_mod.HAS_OTLP
        tracing_mod.HAS_OTLP = False
        try:
            configure_tracing(exporter="otlp")
            assert get_tracer() is None
        finally:
            tracing_mod.HAS_OTLP = original

    def test_otlp_with_otlp_package_creates_tracer(self) -> None:
        """OTLP exporter sets up a tracer when both packages are installed."""
        if not tracing_mod.HAS_OTEL or not tracing_mod.HAS_OTLP:
            pytest.skip("opentelemetry + otlp exporter not installed")
        configure_tracing(exporter="otlp", endpoint="http://localhost:4317")
        assert get_tracer() is not None

    def test_no_otel_installed_disables_tracer(self) -> None:
        """When HAS_OTEL is False, configure_tracing always disables."""
        original = tracing_mod.HAS_OTEL
        tracing_mod.HAS_OTEL = False
        try:
            configure_tracing(exporter="console")
            assert get_tracer() is None
        finally:
            tracing_mod.HAS_OTEL = original


@pytest.mark.unit
class TestTracedLLMCallNoTracer:
    """Tests for traced_llm_call when tracing is disabled."""

    def setup_method(self) -> None:
        disable_tracing()

    @pytest.mark.asyncio
    async def test_yields_empty_dict(self) -> None:
        async with traced_llm_call(model="test", provider="fake") as span_data:
            assert isinstance(span_data, dict)
            assert len(span_data) == 0

    @pytest.mark.asyncio
    async def test_span_data_is_writable(self) -> None:
        async with traced_llm_call(model="test", provider="fake") as span_data:
            span_data["response"] = "something"
        assert span_data["response"] == "something"

    @pytest.mark.asyncio
    async def test_exception_propagates(self) -> None:
        with pytest.raises(ValueError, match="boom"):
            async with traced_llm_call(model="test", provider="fake"):
                raise ValueError("boom")


@pytest.mark.unit
class TestTracedLLMCallWithTracer:
    """Tests for traced_llm_call when tracing is active (mocked tracer)."""

    def setup_method(self) -> None:
        disable_tracing()

    def teardown_method(self) -> None:
        disable_tracing()

    def _setup_mock_tracer(self) -> tuple[MagicMock, MagicMock]:
        """Install a mock tracer and return (tracer, span)."""
        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_span)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_ctx

        tracing_mod._tracer = mock_tracer
        return mock_tracer, mock_span

    @pytest.mark.asyncio
    async def test_creates_span_with_model_and_provider(self) -> None:
        mock_tracer, mock_span = self._setup_mock_tracer()

        async with traced_llm_call(model="claude-3", provider="anthropic"):
            pass

        mock_tracer.start_as_current_span.assert_called_once_with("llm.complete")
        mock_span.set_attribute.assert_any_call("llm.model", "claude-3")
        mock_span.set_attribute.assert_any_call("llm.provider", "anthropic")

    @pytest.mark.asyncio
    async def test_custom_operation_name(self) -> None:
        mock_tracer, _ = self._setup_mock_tracer()

        async with traced_llm_call(model="test", provider="test", operation="custom.op"):
            pass

        mock_tracer.start_as_current_span.assert_called_once_with("custom.op")

    @pytest.mark.asyncio
    async def test_sets_response_attributes_on_span(self) -> None:
        _, mock_span = self._setup_mock_tracer()

        usage = TokenUsage(
            input_tokens=100, output_tokens=50, input_cost_usd=0.001, output_cost_usd=0.002
        )
        response = LLMResponse(
            content="test",
            usage=usage,
            model="test-model",
            provider="test",
            latency_ms=42.5,
        )

        async with traced_llm_call(model="test-model", provider="test") as span_data:
            span_data["response"] = response

        mock_span.set_attribute.assert_any_call("llm.input_tokens", 100)
        mock_span.set_attribute.assert_any_call("llm.output_tokens", 50)
        mock_span.set_attribute.assert_any_call("llm.total_tokens", 150)
        mock_span.set_attribute.assert_any_call("llm.cost_usd", 0.003)
        mock_span.set_attribute.assert_any_call("llm.latency_ms", 42.5)

    @pytest.mark.asyncio
    async def test_no_response_skips_attributes(self) -> None:
        _, mock_span = self._setup_mock_tracer()

        async with traced_llm_call(model="test", provider="test"):
            pass

        # Only model and provider should be set, not token attributes
        calls = [c[0] for c in mock_span.set_attribute.call_args_list]
        attr_names = [c[0] for c in calls]
        assert "llm.model" in attr_names
        assert "llm.provider" in attr_names
        assert "llm.input_tokens" not in attr_names

    @pytest.mark.asyncio
    async def test_exception_records_error_on_span(self) -> None:
        if not tracing_mod.HAS_OTEL:
            pytest.skip("opentelemetry not installed")

        _, mock_span = self._setup_mock_tracer()

        with pytest.raises(RuntimeError, match="fail"):
            async with traced_llm_call(model="test", provider="test"):
                raise RuntimeError("fail")

        mock_span.set_status.assert_called_once()
        mock_span.record_exception.assert_called_once()
