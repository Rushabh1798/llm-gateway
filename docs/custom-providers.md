# Custom Providers

## Step-by-Step Guide

1. Create a class implementing the `LLMProvider` protocol
2. Add a `from_config(cls, config)` classmethod
3. Register with `register_provider("name", YourProvider.from_config)`
4. Use via `GatewayConfig(provider="name")`

See `examples/custom_provider.py` for a complete example.

::: llm_gateway.registry
