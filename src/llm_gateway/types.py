"""Core data types for llm-gateway."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Literal, TypedDict, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class TokenUsage:
    """Token counts and associated costs for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed."""
        return self.input_tokens + self.output_tokens

    @property
    def total_cost_usd(self) -> float:
        """Total cost in USD."""
        return self.input_cost_usd + self.output_cost_usd


@dataclass
class LLMResponse(Generic[T]):
    """Standardized response wrapper from any LLM provider.

    Generic over T — the validated Pydantic model type returned in `content`.
    """

    content: T
    usage: TokenUsage
    model: str
    provider: str
    latency_ms: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


class LLMMessage(TypedDict, total=False):
    """A single message in the conversation.

    Compatible with Anthropic and OpenAI message formats.
    """

    role: Literal["user", "assistant", "system"]
    content: str
