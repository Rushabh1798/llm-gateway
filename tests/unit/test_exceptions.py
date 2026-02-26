"""Tests for exception hierarchy."""

from __future__ import annotations

import pytest

from llm_gateway.exceptions import (
    CLINotFoundError,
    CostLimitExceededError,
    GatewayError,
    ProviderError,
    ProviderNotFoundError,
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
