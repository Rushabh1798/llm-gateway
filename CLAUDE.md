# llm-gateway

## Project Overview

Production-ready, vendor-agnostic LLM gateway Python package. Consumers import `LLMClient`, configure via `LLM_*` env vars, and switch providers with zero code changes.

## Architecture

```
Consumer → LLMClient → Provider (via Registry) → LLM API / CLI
                ↕              ↕
          CostTracker    OTEL Tracer
```

- **Single import**: `from llm_gateway import LLMClient` is the only import consumers need.
- **Config-driven**: `GatewayConfig(BaseSettings)` reads `LLM_*` env vars. Change `.env`, restart — done.
- **Protocol-based providers**: `LLMProvider` Protocol with `complete()` and `close()`. Providers are registered via factory functions in a lazy registry.
- **Structured output**: Every `complete()` call takes a `response_model: type[T]` (Pydantic BaseModel) and returns `LLMResponse[T]` with validated content.
- **Cost tracking**: `CostTracker` accumulates token usage and USD cost per client instance. Supports warn threshold and hard limit.
- **Observability**: OpenTelemetry spans per LLM call; structlog with JSON/console rendering.

## Source Map

| File | Purpose |
|------|---------|
| `src/llm_gateway/__init__.py` | Public API (19 exports in `__all__`) |
| `src/llm_gateway/client.py` | `LLMClient` — the single class consumers use |
| `src/llm_gateway/config.py` | `GatewayConfig(BaseSettings)` with `LLM_` prefix |
| `src/llm_gateway/types.py` | `TokenUsage` (frozen), `LLMResponse[T]`, `LLMMessage` |
| `src/llm_gateway/exceptions.py` | `GatewayError` hierarchy (7 exception classes) |
| `src/llm_gateway/cost.py` | `_PRICING` registry, `CostTracker`, `calculate_cost()` |
| `src/llm_gateway/registry.py` | `register_provider()`, `build_provider()`, lazy builtins |
| `src/llm_gateway/providers/base.py` | `LLMProvider` Protocol |
| `src/llm_gateway/providers/anthropic.py` | `AnthropicProvider` (instructor + AsyncAnthropic) |
| `src/llm_gateway/providers/local_claude.py` | `LocalClaudeProvider` (claude CLI subprocess) |
| `src/llm_gateway/observability/tracing.py` | OTEL setup, `traced_llm_call` context manager |
| `src/llm_gateway/observability/logging.py` | structlog / stdlib fallback |

## Testing

| Suite | Location | Count | Command |
|-------|----------|-------|---------|
| Unit tests | `tests/unit/` | 39 | `pytest -m unit -v` |
| Integration (dry-run) | `integration_tests/tests/test_dry_run.py` | 22 | `cd integration_tests && pytest -v` |
| Integration (live) | `integration_tests/tests/test_live.py` | 10 | `cd integration_tests && pytest --run-live -m live -v` |

Integration tests are an independent Python project under `integration_tests/` that installs llm-gateway as a dependency (not direct source import). In CI, llm-gateway is installed from the checkout; locally, it uses a `file://` reference.

Live tests print a session summary: total Claude CLI calls, estimated tokens, and cost.

## Key Decisions

1. **CLAUDECODE env var stripping**: `LocalClaudeProvider._run_cli()` strips `CLAUDECODE` from the subprocess environment to allow running inside a Claude Code session.
2. **Heuristic token estimation**: Local claude uses ~4 chars/token heuristic since the CLI doesn't report actual tokens. Cost is always $0 (no API fees).
3. **Integration test isolation**: `integration_tests/` has its own `pyproject.toml` and installs llm-gateway as a package dependency. This validates the public API surface and package installability.
4. **Pre-commit mirrors CI**: Hooks run ruff, mypy, unit tests, and integration dry-run tests — same as the CI pipeline.
5. **Optional heavy deps**: Anthropic SDK, OpenTelemetry, and structlog are optional extras. Core package has only pydantic, pydantic-settings, and tenacity.

## Development Commands

```bash
pip install -e ".[dev]"                    # install with all dev deps
pre-commit install                         # set up git hooks

ruff check . && ruff format --check .      # lint
mypy .                                     # type check
pytest -m unit -v                          # unit tests
cd integration_tests && pytest -v          # integration dry-run
cd integration_tests && pytest --run-live -m live -v  # live tests (requires claude CLI)
pre-commit run --all-files                 # all pre-commit checks
```
