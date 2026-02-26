# llm-gateway

[![CI](https://github.com/YOUR_ORG/llm-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_ORG/llm-gateway/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llm-gateway)](https://pypi.org/project/llm-gateway/)
[![Python](https://img.shields.io/pypi/pyversions/llm-gateway)](https://pypi.org/project/llm-gateway/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-ready, vendor-agnostic LLM gateway** with config-driven provider switching, built-in cost tracking, and OpenTelemetry observability.

## Features

- **Zero-code provider switching** — Change `LLM_PROVIDER` in `.env`, restart. Done.
- **Structured output** — Every call returns a validated Pydantic model via `response_model`.
- **Built-in cost tracking** — Token usage and USD cost on every response; configurable guardrails.
- **Observability** — OpenTelemetry spans per LLM call with model, tokens, cost, latency attributes.
- **Extensible** — Add custom providers with a single factory function.
- **Type-safe** — Full type annotations, `py.typed`, strict mypy.

## Quick Start

```bash
pip install 'llm-gateway[anthropic]'
```

```python
import asyncio
from pydantic import BaseModel
from llm_gateway import LLMClient

class Answer(BaseModel):
    text: str

async def main():
    llm = LLMClient()  # reads LLM_* env vars
    resp = await llm.complete(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        response_model=Answer,
    )
    print(resp.content.text)        # "4"
    print(resp.usage.total_cost_usd)  # 0.000123
    await llm.close()

asyncio.run(main())
```

## Configuration

All settings use the `LLM_` prefix and are read from environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | Provider: `anthropic`, `local_claude`, `openai` |
| `LLM_MODEL` | `claude-sonnet-4-5-20250514` | Model identifier |
| `LLM_API_KEY` | — | API key (falls back to `ANTHROPIC_API_KEY` etc.) |
| `LLM_MAX_TOKENS` | `4096` | Max response tokens |
| `LLM_MAX_RETRIES` | `3` | Retry attempts |
| `LLM_TIMEOUT_SECONDS` | `120` | Request timeout |
| `LLM_COST_LIMIT_USD` | — | Hard cost limit per client instance |
| `LLM_COST_WARN_USD` | — | Warning threshold |
| `LLM_TRACE_ENABLED` | `false` | Enable OTEL tracing |
| `LLM_TRACE_EXPORTER` | `none` | `none`, `console`, `otlp` |
| `LLM_LOG_LEVEL` | `INFO` | Log level |
| `LLM_LOG_FORMAT` | `json` | `json` or `console` |

## Providers

### Anthropic (default)

```bash
pip install 'llm-gateway[anthropic]'
```

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

### Local Claude CLI

Use the Claude Code CLI for local inference — no API key needed:

```env
LLM_PROVIDER=local_claude
```

### Adding a Custom Provider

```python
from llm_gateway import register_provider, LLMProvider

class MyProvider:
    async def complete(self, messages, response_model, model, **kwargs):
        ...  # Your implementation
    async def close(self):
        ...

register_provider("my_provider", lambda config: MyProvider())
```

## Cost Tracking

Every response includes token usage and cost:

```python
resp = await llm.complete(messages, response_model=MyModel)
print(resp.usage.input_tokens)    # 150
print(resp.usage.output_tokens)   # 42
print(resp.usage.total_cost_usd)  # 0.000654

# Cumulative across all calls
print(llm.total_cost_usd)  # 0.001234
print(llm.cost_summary())  # {"total_tokens": ..., "total_cost_usd": ..., ...}
```

## License

MIT
