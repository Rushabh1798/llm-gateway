"""Exception hierarchy for llm-gateway."""

from __future__ import annotations


class GatewayError(Exception):
    """Base exception for all llm-gateway errors."""


class ProviderNotFoundError(GatewayError):
    """Raised when the requested provider is not registered."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(
            f"Provider '{provider}' is not registered. "
            f"Check LLM_PROVIDER env var and installed extras."
        )


class ProviderInitError(GatewayError):
    """Raised when a provider fails to initialize."""

    def __init__(self, provider: str, reason: str) -> None:
        self.provider = provider
        super().__init__(f"Failed to initialize provider '{provider}': {reason}")


class ProviderError(GatewayError):
    """Raised when the underlying provider SDK raises an error."""

    def __init__(self, provider: str, original: Exception) -> None:
        self.provider = provider
        self.original = original
        super().__init__(f"Provider '{provider}' error: {original}")


class CostLimitExceededError(GatewayError):
    """Raised when cumulative cost exceeds the configured limit."""

    def __init__(self, current: float, limit: float) -> None:
        self.current = current
        self.limit = limit
        super().__init__(
            f"Cost limit exceeded: ${current:.4f} >= ${limit:.4f}. "
            f"Increase LLM_COST_LIMIT_USD or create a new LLMClient."
        )


class ResponseValidationError(GatewayError):
    """Raised when the LLM response cannot be validated against the model."""

    def __init__(self, model_name: str, reason: str) -> None:
        self.model_name = model_name
        super().__init__(
            f"Failed to validate response as {model_name}: {reason}"
        )


class CLINotFoundError(ProviderError):
    """Raised when the claude CLI binary is not found in PATH."""

    def __init__(self) -> None:
        super().__init__(
            provider="local_claude",
            original=FileNotFoundError(
                "'claude' CLI not found in PATH. "
                "Install it: https://docs.anthropic.com/en/docs/claude-code"
            ),
        )
