"""LLM provider protocol — the contract every provider must satisfy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar, runtime_checkable

from llm_gateway.types import LLMMessage, LLMResponse

T = TypeVar("T")


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol that all LLM providers must implement.

    Providers handle the actual communication with the LLM service
    and return standardized LLMResponse objects.
    """

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[T],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse[T]:
        """Send messages to the LLM and return a structured response.

        Args:
            messages: Conversation messages.
            response_model: Pydantic model class for structured output.
            model: Model identifier. ``None`` means the provider picks its default.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with validated content, token usage, and cost.
        """
        ...

    async def close(self) -> None:
        """Clean up provider resources (HTTP sessions, subprocesses, etc.)."""
        ...
