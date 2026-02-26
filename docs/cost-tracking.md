# Cost Tracking

## TokenUsage

Every `LLMResponse` includes a `TokenUsage` with input/output token counts and USD costs.

## CostTracker

`LLMClient` uses `CostTracker` internally to accumulate costs across calls.

## Guardrails

- `LLM_COST_WARN_USD` — emit a warning when cumulative cost exceeds this threshold
- `LLM_COST_LIMIT_USD` — raise `CostLimitExceededError` when exceeded

## Pricing Registry

Use `register_pricing()` to add custom model pricing.

::: llm_gateway.cost
