"""llm-gateway — Production-ready, vendor-agnostic LLM gateway.

Usage:
    from llm_gateway import LLMClient, LLMResponse, GatewayConfig

    llm = LLMClient()  # reads LLM_* env vars
    resp = await llm.complete(messages, response_model=MyModel)
"""

from __future__ import annotations

from llm_gateway.client import LLMClient
from llm_gateway.config import GatewayConfig
from llm_gateway.cost import CostTracker, calculate_cost, register_pricing
from llm_gateway.exceptions import (
    CLINotFoundError,
    CostLimitExceededError,
    GatewayError,
    ProviderError,
    ProviderInitError,
    ProviderNotFoundError,
    ResponseValidationError,
)
from llm_gateway.providers.base import LLMProvider
from llm_gateway.registry import build_provider, list_providers, register_provider
from llm_gateway.types import LLMMessage, LLMResponse, TokenUsage

__all__ = [
    # Core
    "LLMClient",
    "GatewayConfig",
    # Types
    "LLMResponse",
    "LLMMessage",
    "TokenUsage",
    # Provider
    "LLMProvider",
    "register_provider",
    "build_provider",
    "list_providers",
    # Cost
    "CostTracker",
    "calculate_cost",
    "register_pricing",
    # Exceptions
    "GatewayError",
    "ProviderNotFoundError",
    "ProviderInitError",
    "ProviderError",
    "CostLimitExceededError",
    "ResponseValidationError",
    "CLINotFoundError",
]
