"""Tests for the provider registry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm_gateway.config import GatewayConfig
from llm_gateway.exceptions import ProviderNotFoundError
from llm_gateway.registry import (
    _PROVIDERS,
    build_provider,
    list_providers,
    register_provider,
)


@pytest.mark.unit
class TestRegistry:
    def test_register_and_build(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_provider = MagicMock()
        factory = MagicMock(return_value=mock_provider)

        register_provider("test_provider", factory)

        monkeypatch.setenv("LLM_PROVIDER", "test_provider")
        config = GatewayConfig()
        result = build_provider(config)

        factory.assert_called_once_with(config)
        assert result is mock_provider

        # Cleanup
        _PROVIDERS.pop("test_provider", None)

    def test_unknown_provider_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "nonexistent_xyz_999")
        config = GatewayConfig()
        with pytest.raises(ProviderNotFoundError, match="nonexistent_xyz_999"):
            build_provider(config)

    def test_list_providers(self) -> None:
        providers = list_providers()
        assert isinstance(providers, list)
        # At minimum, anthropic should be available if installed
