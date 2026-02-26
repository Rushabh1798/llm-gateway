# Observability

## OpenTelemetry Tracing

Set `LLM_TRACE_ENABLED=true` and `LLM_TRACE_EXPORTER=console` (or `otlp`) to enable tracing.

Spans include: `llm.model`, `llm.provider`, `llm.input_tokens`, `llm.output_tokens`, `llm.cost_usd`, `llm.latency_ms`.

## Structured Logging

If `structlog` is installed, llm-gateway uses structured logging with JSON or console output.

Set `LLM_LOG_FORMAT=console` for development-friendly output.

::: llm_gateway.observability
