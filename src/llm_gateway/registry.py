"""Provider registry — maps provider names to factory functions."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from llm_gateway.exceptions import ProviderInitError, ProviderNotFoundError

if TYPE_CHECKING:
    from llm_gateway.config import GatewayConfig
    from llm_gateway.providers.base import LLMProvider

logger = logging.getLogger(__name__)

# Global registry: name → factory(config) → provider instance
_PROVIDERS: dict[str, Callable[["GatewayConfig"], "LLMProvider"]] = {}


def register_provider(
    name: str,
    factory: Callable[["GatewayConfig"], "LLMProvider"],
) -> None:
    """Register a provider factory.

    Args:
        name: Provider name (e.g. "anthropic", "local_claude", "openai").
        factory: Callable that takes GatewayConfig and returns an LLMProvider.
    """
    _PROVIDERS[name] = factory
    logger.debug("Registered LLM provider: %s", name)


def build_provider(config: "GatewayConfig") -> "LLMProvider":
    """Build a provider instance from configuration.

    Triggers lazy registration of built-in providers on first call.

    Args:
        config: Gateway configuration with provider name and settings.

    Returns:
        An initialized LLMProvider instance.

    Raises:
        ProviderNotFoundError: If the provider name is not registered.
        ProviderInitError: If the provider factory raises an error.
    """
    _ensure_builtins_registered()

    factory = _PROVIDERS.get(config.provider)
    if factory is None:
        raise ProviderNotFoundError(config.provider)

    try:
        return factory(config)
    except Exception as exc:
        raise ProviderInitError(config.provider, str(exc)) from exc


def list_providers() -> list[str]:
    """Return names of all registered providers."""
    _ensure_builtins_registered()
    return list(_PROVIDERS.keys())


# ── Lazy Registration ───────────────────────────────────────────

_builtins_registered = False


def _ensure_builtins_registered() -> None:
    """Lazily register built-in providers on first use.

    This avoids importing heavy SDKs (anthropic, openai) at module load time.
    Import errors are caught — providers for uninstalled SDKs are simply
    not registered.
    """
    global _builtins_registered  # noqa: PLW0603
    if _builtins_registered:
        return
    _builtins_registered = True

    # Anthropic
    try:
        from llm_gateway.providers.anthropic import AnthropicProvider

        register_provider("anthropic", AnthropicProvider.from_config)
    except ImportError:
        logger.debug("anthropic extras not installed — provider not available")

    # Local Claude CLI
    try:
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        register_provider("local_claude", LocalClaudeProvider.from_config)
    except ImportError:
        logger.debug("local_claude provider not available")

    # OpenAI (future)
    try:
        from llm_gateway.providers.openai import OpenAIProvider  # type: ignore[import-not-found]

        register_provider("openai", OpenAIProvider.from_config)
    except ImportError:
        logger.debug("openai extras not installed — provider not available")
