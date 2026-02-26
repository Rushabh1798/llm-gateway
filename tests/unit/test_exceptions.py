"""Tests for exception hierarchy."""

from __future__ import annotations

import pytest

from llm_gateway.exceptions import (
    CLINotFoundError,
    CostLimitExceededError,
    GatewayError,
    ProviderError,
    ProviderInitError,
    ProviderNotFoundError,
    ResponseValidationError,
)


@pytest.mark.unit
class TestExceptions:
    def test_hierarchy(self) -> None:
        assert issubclass(ProviderNotFoundError, GatewayError)
        assert issubclass(CostLimitExceededError, GatewayError)
        assert issubclass(ProviderError, GatewayError)
        assert issubclass(CLINotFoundError, ProviderError)

    def test_provider_not_found_message(self) -> None:
        exc = ProviderNotFoundError("foobar")
        assert "foobar" in str(exc)
        assert exc.provider == "foobar"

    def test_cost_limit_exceeded(self) -> None:
        exc = CostLimitExceededError(current=5.0, limit=3.0)
        assert exc.current == 5.0
        assert exc.limit == 3.0
        assert "$5.0" in str(exc) or "5.0000" in str(exc)

    def test_provider_init_error(self) -> None:
        exc = ProviderInitError("anthropic", "missing API key")
        assert exc.provider == "anthropic"
        assert "anthropic" in str(exc)
        assert "missing API key" in str(exc)

    def test_provider_error(self) -> None:
        original = RuntimeError("connection timeout")
        exc = ProviderError("openai", original)
        assert exc.provider == "openai"
        assert exc.original is original
        assert "openai" in str(exc)
        assert "connection timeout" in str(exc)

    def test_response_validation_error(self) -> None:
        exc = ResponseValidationError("MyModel", "field 'name' is required")
        assert exc.model_name == "MyModel"
        assert "MyModel" in str(exc)
        assert "field 'name' is required" in str(exc)

    def test_cli_not_found_error(self) -> None:
        exc = CLINotFoundError()
        assert exc.provider == "local_claude"
        assert isinstance(exc.original, FileNotFoundError)
        assert "claude" in str(exc).lower()
