# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Initial release
- `LLMClient` with config-driven provider selection
- `AnthropicProvider` — Anthropic API via instructor
- `LocalClaudeProvider` — Claude Code CLI subprocess
- `GatewayConfig` with `LLM_*` environment variable support
- Token usage tracking and cost estimation
- Cost guardrails (warning threshold + hard limit)
- OpenTelemetry tracing (none/console/otlp exporters)
- Structured logging via structlog
- Provider registry with lazy loading
- `FakeLLMProvider` for testing
- Full test suite (unit + integration)
- CI/CD with GitHub Actions
- MkDocs documentation
