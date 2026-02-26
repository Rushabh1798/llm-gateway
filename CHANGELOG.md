# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- `FakeLLMProvider` shipped as public testing utility (`llm_gateway.testing`)
- `fake` provider registered in provider registry (`LLM_PROVIDER=fake`)
- `FakeCall` dataclass for inspecting recorded calls in tests
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
- Full unit test suite (39 tests, 82%+ coverage)
- Integration test suite as independent consumer project (`integration_tests/`)
  - 22 dry-run tests (mocked, no real LLM calls)
  - 10 live tests (real Claude CLI calls with structured output validation)
  - Live test suite summary: CLI sessions, token usage, cost estimation
- CI/CD with GitHub Actions (lint, unit tests, integration tests, minimal-deps)
- Pre-commit hooks mirroring CI (ruff, mypy, unit tests, integration dry-run)
- Git dependency support (install via commit SHA)
- MkDocs documentation stubs

### Fixed
- `LocalClaudeProvider` strips `CLAUDECODE` env var to allow nested CLI sessions
