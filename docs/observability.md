# Observability

## OpenTelemetry Tracing

Set `LLM_TRACE_ENABLED=true` and `LLM_TRACE_EXPORTER=console` (or `otlp`) to enable tracing.

Spans include: `llm.model`, `llm.provider`, `llm.input_tokens`, `llm.output_tokens`, `llm.cost_usd`, `llm.latency_ms`.

## Structured Logging

If `structlog` is installed, llm-gateway uses structured logging with JSON or console output.

Set `LLM_LOG_FORMAT=console` for development-friendly output.

## Provider-Level Logging (Local Claude CLI)

`LocalClaudeProvider` emits structured logs at every stage of a CLI interaction:

| Level | Event | Content |
|-------|-------|---------|
| DEBUG | `claude_cli_request` | Prompt preview (first 500 chars), model, response_model name |
| DEBUG | `claude_cli_raw_result` | Raw CLI output (first 2000 chars), duration, cost from wrapper |
| INFO | `claude_cli_complete` | Latency, estimated tokens, parsed field values (`field=value`) |
| WARNING | `claude_cli_stderr` | Non-empty stderr from the CLI subprocess |
| ERROR | `claude_cli_timeout` | Timeout threshold that was exceeded |
| ERROR | `claude_cli_error` | Exception type and message on CLI failure |

During live integration tests (`pytest --run-live -v`), these logs stream to the console in real time via pytest's live log capture at DEBUG level.

::: llm_gateway.observability
