# Providers

## Built-in Providers

### Anthropic

Uses the Anthropic API via `instructor` for structured output.

```bash
pip install 'llm-gateway[anthropic]'
```

### Local Claude CLI

Delegates to the `claude` CLI binary for local inference. No API key needed.

```env
LLM_PROVIDER=local_claude
```

The provider logs every CLI interaction at DEBUG (prompt, raw response) and INFO (latency, tokens, parsed content) levels. See [Observability](observability.md#provider-level-logging-local-claude-cli) for the full log event table.

## Provider Protocol

All providers implement the `LLMProvider` protocol defined in `providers/base.py`.

::: llm_gateway.providers.base.LLMProvider
