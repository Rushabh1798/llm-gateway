"""OpenTelemetry tracing for LLM calls."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from llm_gateway.types import LLMResponse

logger = logging.getLogger(__name__)

# ── Optional OTEL imports ───────────────────────────────────────
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )

    HAS_OTLP = True
except ImportError:
    HAS_OTLP = False


# Module-level tracer (NoOp if OTEL not installed/configured)
_tracer: Any = None


def configure_tracing(
    exporter: str = "none",
    endpoint: str = "http://localhost:4317",
    service_name: str = "llm-gateway",
) -> None:
    """Configure OpenTelemetry tracing.

    Args:
        exporter: One of "none", "console", "otlp".
        endpoint: OTLP collector endpoint (only used when exporter="otlp").
        service_name: Service name for spans.
    """
    global _tracer

    if exporter == "none" or not HAS_OTEL:
        _tracer = None
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if exporter == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif exporter == "otlp":
        if not HAS_OTLP:
            logger.warning("OTLP exporter requested but opentelemetry-exporter-otlp not installed")
            _tracer = None
            return
        provider.add_span_processor(
            SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
        )

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("llm_gateway")
    logger.info("OTEL tracing configured: exporter=%s, service=%s", exporter, service_name)


def get_tracer() -> Any:
    """Return the configured tracer, or None if tracing is disabled."""
    return _tracer


def disable_tracing() -> None:
    """Disable tracing (useful for tests)."""
    global _tracer
    _tracer = None


@asynccontextmanager
async def traced_llm_call(
    model: str | None,
    provider: str,
    operation: str = "llm.complete",
) -> AsyncGenerator[dict[str, Any], None]:
    """Context manager that creates an OTEL span for an LLM call.

    Usage:
        async with traced_llm_call("claude-sonnet", "anthropic") as span_attrs:
            response = await provider.complete(...)
            span_attrs["response"] = response  # set after call

    The span will automatically record:
    - llm.model, llm.provider
    - llm.input_tokens, llm.output_tokens, llm.cost_usd, llm.latency_ms
    - Error status if an exception is raised
    """
    span_data: dict[str, Any] = {}

    if _tracer is None:
        yield span_data
        return

    with _tracer.start_as_current_span(operation) as span:
        span.set_attribute("llm.model", model or "provider-default")
        span.set_attribute("llm.provider", provider)

        try:
            yield span_data
        except Exception as exc:
            span.set_status(trace.StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise
        else:
            # Extract response data if provided
            response = span_data.get("response")
            if response is not None and isinstance(response, LLMResponse):
                span.set_attribute("llm.input_tokens", response.usage.input_tokens)
                span.set_attribute("llm.output_tokens", response.usage.output_tokens)
                span.set_attribute("llm.total_tokens", response.usage.total_tokens)
                span.set_attribute("llm.cost_usd", response.usage.total_cost_usd)
                span.set_attribute("llm.latency_ms", response.latency_ms)
