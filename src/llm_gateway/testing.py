"""Testing utilities shipped with llm-gateway.

Provides ``FakeLLMProvider`` for consumers to use in their test suites
without reimplementing the LLMProvider Protocol.

Usage::

    from llm_gateway import LLMClient, FakeLLMProvider
    from pydantic import BaseModel

    class Answer(BaseModel):
        text: str

    fake = FakeLLMProvider()
    fake.set_response(Answer, Answer(text="42"))

    async with LLMClient(provider_instance=fake) as client:
        resp = await client.complete(
            messages=[{"role": "user", "content": "What is 6*7?"}],
            response_model=Answer,
        )
        assert resp.content.text == "42"
        assert fake.call_count == 1
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from llm_gateway.cost import build_token_usage
from llm_gateway.exceptions import ResponseValidationError
from llm_gateway.types import LLMMessage, LLMResponse

if TYPE_CHECKING:
    from llm_gateway.config import GatewayConfig

T = TypeVar("T")


@dataclass
class FakeCall:
    """Record of a single ``FakeLLMProvider.complete()`` invocation."""

    messages: Sequence[LLMMessage]
    response_model: type
    model: str
    response: object


class FakeLLMProvider:
    """Fake LLM provider for testing. Implements the ``LLMProvider`` Protocol.

    Two modes:

    1. **Pre-configured**: call ``set_response(ModelClass, instance)`` before
       ``complete()``.
    2. **Dynamic**: pass a ``response_factory`` callable to the constructor.

    Resolution order in ``complete()``:

    1. Pre-configured via ``set_response()`` (exact type match)
    2. ``response_factory`` callable (if provided)
    3. Raise ``ResponseValidationError`` (no response configured)
    """

    def __init__(
        self,
        response_factory: Callable[[type[T], Sequence[LLMMessage]], T] | None = None,
        default_input_tokens: int = 100,
        default_output_tokens: int = 50,
    ) -> None:
        self._responses: dict[type, object] = {}
        self._response_factory = response_factory
        self._default_input_tokens = default_input_tokens
        self._default_output_tokens = default_output_tokens
        self.calls: list[FakeCall] = []

    def set_response(self, response_model: type[T], response: T) -> None:
        """Pre-configure a response for a specific ``response_model`` class."""
        self._responses[response_model] = response

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[T],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse[T]:
        """Return pre-configured or factory-built response.

        Args:
            messages: Conversation messages.
            response_model: Pydantic model class for structured output.
            model: Model identifier.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.

        Returns:
            ``LLMResponse[T]`` with configurable ``TokenUsage``.

        Raises:
            ResponseValidationError: If no response is configured and no factory
                is provided.
        """
        content: T | None = None

        # 1. Pre-configured response (exact type match)
        preconfigured = self._responses.get(response_model)
        if preconfigured is not None:
            content = preconfigured  # type: ignore[assignment]

        # 2. response_factory callable
        if content is None and self._response_factory is not None:
            content = self._response_factory(response_model, messages)  # type: ignore[assignment,arg-type]

        # 3. No response available
        if content is None:
            raise ResponseValidationError(
                model_name=response_model.__name__,
                reason="No fake response configured. "
                "Use set_response() or pass a response_factory.",
            )

        usage = build_token_usage(
            model,
            self._default_input_tokens,
            self._default_output_tokens,
        )

        response = LLMResponse(
            content=content,
            usage=usage,
            model=model,
            provider="fake",
            latency_ms=0.0,
        )

        self.calls.append(
            FakeCall(
                messages=messages,
                response_model=response_model,
                model=model,
                response=content,
            )
        )

        return response

    @property
    def call_count(self) -> int:
        """Number of ``complete()`` calls recorded."""
        return len(self.calls)

    async def close(self) -> None:
        """No-op cleanup."""

    @classmethod
    def from_config(cls, config: GatewayConfig) -> FakeLLMProvider:
        """Factory for provider registry. Creates an empty ``FakeLLMProvider``."""
        return cls()
