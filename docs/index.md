# llm-gateway

Production-ready, vendor-agnostic LLM gateway with config-driven provider
switching, built-in cost tracking, and OpenTelemetry observability.

## Why llm-gateway?

- **One import, all providers** — `from llm_gateway import LLMClient`
- **Switch providers via `.env`** — no code changes, ever
- **Know what you spend** — token usage and USD cost on every response
- **See what happens** — OTEL spans with model, tokens, cost, latency
- **Stay safe** — cost guardrails prevent runaway spending
- **Stay typed** — full type annotations, strict mypy, `py.typed`

## Installation

```bash
pip install 'llm-gateway[anthropic]'    # Anthropic provider
pip install 'llm-gateway[all]'          # All providers + tracing + logging
pip install 'llm-gateway[dev]'          # Development dependencies
```
