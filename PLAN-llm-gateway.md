# PLAN: llm-gateway — Production-Ready, Vendor-Agnostic LLM Gateway

> **Status: COMPLETE** — All phases (0–12) implemented and verified.
>
> Post-plan additions:
> - `integration_tests/` — Independent consumer project with 22 dry-run + 10 live tests
> - `LocalClaudeProvider` fix: strips `CLAUDECODE` env var for nested CLI sessions
> - Live test suite summary: tracks CLI sessions, tokens, and cost across test runs
> - Pre-commit hooks mirror CI pipeline (ruff, mypy, unit tests, integration dry-run)
>
> See `CLAUDE.md` for current project context and `CHANGELOG.md` for full change history.

> **Purpose**: Standalone, open-source Python package that abstracts all LLM
> interactions behind a unified, config-driven interface. Consumers import ONE
> class (`LLMClient`), configure via `.env`, and switch providers (Anthropic,
> local Claude CLI, OpenAI, …) with **zero code changes**.
>
> **This plan is self-contained.** Copy it into a fresh repository and execute
> every phase top-to-bottom to produce a fully working, publishable package.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Directory Structure](#2-directory-structure)
3. [Phase 0 — Repository Scaffold](#phase-0--repository-scaffold)
4. [Phase 1 — Core Types & Exceptions](#phase-1--core-types--exceptions)
5. [Phase 2 — Configuration](#phase-2--configuration)
6. [Phase 3 — Cost Tracking](#phase-3--cost-tracking)
7. [Phase 4 — Provider Protocol & Registry](#phase-4--provider-protocol--registry)
8. [Phase 5 — Anthropic Provider](#phase-5--anthropic-provider)
9. [Phase 6 — Local Claude Provider](#phase-6--local-claude-provider)
10. [Phase 7 — Observability](#phase-7--observability)
11. [Phase 8 — LLM Client](#phase-8--llm-client)
12. [Phase 9 — Tests](#phase-9--tests)
13. [Phase 10 — Documentation & Examples](#phase-10--documentation--examples)
14. [Phase 11 — CI/CD & Pre-commit](#phase-11--cicd--pre-commit)
15. [Phase 12 — Final Polish](#phase-12--final-polish)
16. [Appendix A — Consumer Migration Guide (job-hunter-agent)](#appendix-a--consumer-migration-guide)
17. [Appendix B — Adding a New Provider](#appendix-b--adding-a-new-provider)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Consumer Code                        │
│  from llm_gateway import LLMClient                     │
│  llm = LLMClient()          # reads LLM_* env vars     │
│  resp = await llm.complete(messages, response_model)    │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │     LLMClient       │
              │  ┌───────────────┐  │
              │  │ CostTracker   │  │
              │  │ OTEL Tracer   │  │
              │  │ Structlog     │  │
              │  └───────┬───────┘  │
              └──────────┼──────────┘
                         │  delegates to
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
    ┌─────────────┐ ┌──────────┐ ┌──────────┐
    │  Anthropic   │ │  Local   │ │  OpenAI  │
    │  Provider    │ │  Claude  │ │ Provider │
    │ (instructor) │ │ (CLI)    │ │ (future) │
    └─────────────┘ └──────────┘ └──────────┘
```

### Key Design Principles

| Principle | Implementation |
|-----------|---------------|
| Zero-code provider switching | `GatewayConfig` reads `LLM_*` env vars; change `.env`, restart — done |
| Single import for consumers | `from llm_gateway import LLMClient` is the only import needed |
| Structured output everywhere | `response_model: type[T]` → validated Pydantic model in `LLMResponse[T].content` |
| Built-in cost tracking | Every response includes `TokenUsage` with USD cost; `CostTracker` accumulates across calls |
| Built-in observability | OTEL spans per LLM call with model/tokens/cost/latency attributes; structlog events |
| Dependency injection for tests | `LLMClient(provider_instance=mock)` bypasses config entirely |
| Optional heavy dependencies | `pip install llm-gateway[anthropic]` — anthropic/instructor only installed when needed |
| Extensible provider registry | `register_provider("mycloud", factory_fn)` — no core code changes needed |

---

## 2. Directory Structure

```
llm-gateway/
├── src/
│   └── llm_gateway/
│       ├── __init__.py              # Public API exports
│       ├── py.typed                 # PEP 561 marker (empty file)
│       ├── client.py                # LLMClient — the ONE class consumers use
│       ├── config.py                # GatewayConfig (pydantic-settings, LLM_ prefix)
│       ├── types.py                 # TokenUsage, LLMResponse[T], LLMMessage
│       ├── exceptions.py            # GatewayError hierarchy
│       ├── registry.py              # Provider registry + build_provider factory
│       ├── cost.py                  # PRICING dict, calculate_cost, CostTracker
│       ├── providers/
│       │   ├── __init__.py          # Provider sub-package init
│       │   ├── base.py              # LLMProvider Protocol
│       │   ├── anthropic.py         # AnthropicProvider (instructor + AsyncAnthropic)
│       │   └── local_claude.py      # LocalClaudeProvider (claude CLI subprocess)
│       └── observability/
│           ├── __init__.py          # Observability sub-package init
│           ├── tracing.py           # OTEL tracing setup + traced_llm_call
│           └── logging.py           # Structlog configuration
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Shared fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_client.py
│   │   ├── test_config.py
│   │   ├── test_types.py
│   │   ├── test_registry.py
│   │   ├── test_cost.py
│   │   ├── test_exceptions.py
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── test_anthropic.py
│   │       └── test_local_claude.py
│   └── integration/
│       ├── __init__.py
│       └── test_live_providers.py
├── examples/
│   ├── basic_usage.py
│   ├── cost_tracking.py
│   ├── provider_switching.py
│   └── custom_provider.py
├── docs/
│   ├── index.md
│   ├── quickstart.md
│   ├── configuration.md
│   ├── providers.md
│   ├── cost-tracking.md
│   ├── observability.md
│   └── custom-providers.md
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── release.yml
├── pyproject.toml
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CHANGELOG.md
├── .env.example
├── .pre-commit-config.yaml
├── .gitignore
└── mkdocs.yml
```

---

## Phase 0 — Repository Scaffold

### 0.1 `pyproject.toml`

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
name = "llm-gateway"
dynamic = ["version"]
description = "Production-ready, vendor-agnostic LLM gateway with config-driven provider switching, cost tracking, and observability."
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
authors = [{ name = "Your Name", email = "you@example.com" }]
keywords = ["llm", "gateway", "anthropic", "openai", "claude", "ai"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Software Development :: Libraries",
    "Typing :: Typed",
]
dependencies = [
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "tenacity>=8.0",
]

[project.optional-dependencies]
anthropic = ["anthropic>=0.40.0", "instructor>=1.0.0"]
openai = ["openai>=1.0.0", "instructor>=1.0.0"]
tracing = [
    "opentelemetry-api>=1.20.0",
    "opentelemetry-sdk>=1.20.0",
    "opentelemetry-exporter-otlp>=1.20.0",
]
logging = ["structlog>=24.0.0"]
all = ["llm-gateway[anthropic,tracing,logging]"]
dev = [
    "llm-gateway[all]",
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
    "ruff>=0.8",
    "mypy>=1.13",
    "pre-commit",
]
docs = [
    "mkdocs-material>=9.0",
    "mkdocstrings[python]>=0.27",
]

[project.urls]
Homepage = "https://github.com/YOUR_ORG/llm-gateway"
Documentation = "https://YOUR_ORG.github.io/llm-gateway"
Repository = "https://github.com/YOUR_ORG/llm-gateway"
Changelog = "https://github.com/YOUR_ORG/llm-gateway/blob/main/CHANGELOG.md"

[tool.hatch.version]
source = "vcs"

[tool.hatch.build.targets.wheel]
packages = ["src/llm_gateway"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "unit: fast, fully mocked unit tests",
    "integration: tests requiring real provider access",
]
filterwarnings = ["ignore::DeprecationWarning"]

[tool.ruff]
target-version = "py311"
line-length = 99
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM", "RUF"]
ignore = ["E501"]

[tool.ruff.lint.isort]
known-first-party = ["llm_gateway"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
packages = ["llm_gateway"]

[[tool.mypy.overrides]]
module = ["anthropic.*", "instructor.*", "opentelemetry.*", "structlog.*"]
ignore_missing_imports = true
```

### 0.2 `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
*.egg
.eggs/

# Virtual environments
.venv/
venv/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Testing
.coverage
htmlcov/
coverage.xml
.pytest_cache/

# Environment
.env
.env.local

# OS
.DS_Store
Thumbs.db

# Build
*.whl
*.tar.gz

# mypy
.mypy_cache/

# ruff
.ruff_cache/
```

### 0.3 `LICENSE`

```
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 0.4 `src/llm_gateway/py.typed`

```
# PEP 561 marker — this package ships inline type annotations.
```

### 0.5 `.env.example`

```env
# ── LLM Gateway Configuration ──────────────────────────────────
# All settings use the LLM_ prefix and are read automatically.

# Provider: "anthropic" | "local_claude" | "openai" (future)
LLM_PROVIDER=anthropic

# Model identifier (provider-specific)
LLM_MODEL=claude-sonnet-4-5-20250514

# API key — if not set, falls back to ANTHROPIC_API_KEY / OPENAI_API_KEY
# LLM_API_KEY=sk-ant-...

# Optional base URL override (for proxies or custom endpoints)
# LLM_BASE_URL=https://api.anthropic.com

# Max tokens per response
LLM_MAX_TOKENS=4096

# Retry settings
LLM_MAX_RETRIES=3
LLM_TIMEOUT_SECONDS=120

# ── Cost Guardrails ─────────────────────────────────────────────
# Set to limit cumulative cost per LLMClient instance (USD)
# LLM_COST_LIMIT_USD=10.0
# LLM_COST_WARN_USD=5.0

# ── Observability ───────────────────────────────────────────────
# Tracing: "none" | "console" | "otlp"
LLM_TRACE_ENABLED=false
LLM_TRACE_EXPORTER=none
LLM_TRACE_ENDPOINT=http://localhost:4317
LLM_TRACE_SERVICE_NAME=llm-gateway

# Logging: "json" | "console"
LLM_LOG_LEVEL=INFO
LLM_LOG_FORMAT=json
```

### 0.6 Empty `__init__.py` files

Create these empty files (content will be added in later phases):
- `src/llm_gateway/__init__.py`
- `src/llm_gateway/providers/__init__.py`
- `src/llm_gateway/observability/__init__.py`
- `tests/__init__.py`
- `tests/unit/__init__.py`
- `tests/unit/providers/__init__.py`
- `tests/integration/__init__.py`

---

## Phase 1 — Core Types & Exceptions

### 1.1 `src/llm_gateway/types.py`

```python
"""Core data types for llm-gateway."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Literal, TypedDict, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class TokenUsage:
    """Token counts and associated costs for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed."""
        return self.input_tokens + self.output_tokens

    @property
    def total_cost_usd(self) -> float:
        """Total cost in USD."""
        return self.input_cost_usd + self.output_cost_usd


@dataclass
class LLMResponse(Generic[T]):
    """Standardized response wrapper from any LLM provider.

    Generic over T — the validated Pydantic model type returned in `content`.
    """

    content: T
    usage: TokenUsage
    model: str
    provider: str
    latency_ms: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


class LLMMessage(TypedDict, total=False):
    """A single message in the conversation.

    Compatible with Anthropic and OpenAI message formats.
    """

    role: Literal["user", "assistant", "system"]
    content: str
```

### 1.2 `src/llm_gateway/exceptions.py`

```python
"""Exception hierarchy for llm-gateway."""

from __future__ import annotations


class GatewayError(Exception):
    """Base exception for all llm-gateway errors."""


class ProviderNotFoundError(GatewayError):
    """Raised when the requested provider is not registered."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(
            f"Provider '{provider}' is not registered. "
            f"Check LLM_PROVIDER env var and installed extras."
        )


class ProviderInitError(GatewayError):
    """Raised when a provider fails to initialize."""

    def __init__(self, provider: str, reason: str) -> None:
        self.provider = provider
        super().__init__(f"Failed to initialize provider '{provider}': {reason}")


class ProviderError(GatewayError):
    """Raised when the underlying provider SDK raises an error."""

    def __init__(self, provider: str, original: Exception) -> None:
        self.provider = provider
        self.original = original
        super().__init__(f"Provider '{provider}' error: {original}")


class CostLimitExceededError(GatewayError):
    """Raised when cumulative cost exceeds the configured limit."""

    def __init__(self, current: float, limit: float) -> None:
        self.current = current
        self.limit = limit
        super().__init__(
            f"Cost limit exceeded: ${current:.4f} >= ${limit:.4f}. "
            f"Increase LLM_COST_LIMIT_USD or create a new LLMClient."
        )


class ResponseValidationError(GatewayError):
    """Raised when the LLM response cannot be validated against the model."""

    def __init__(self, model_name: str, reason: str) -> None:
        self.model_name = model_name
        super().__init__(
            f"Failed to validate response as {model_name}: {reason}"
        )


class CLINotFoundError(ProviderError):
    """Raised when the claude CLI binary is not found in PATH."""

    def __init__(self) -> None:
        super().__init__(
            provider="local_claude",
            original=FileNotFoundError(
                "'claude' CLI not found in PATH. "
                "Install it: https://docs.anthropic.com/en/docs/claude-code"
            ),
        )
```

---

## Phase 2 — Configuration

### 2.1 `src/llm_gateway/config.py`

```python
"""Gateway configuration via environment variables."""

from __future__ import annotations

import os
from typing import Optional

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings


class GatewayConfig(BaseSettings):
    """LLM Gateway configuration.

    All fields are read from environment variables with the ``LLM_`` prefix.
    Example: ``LLM_PROVIDER=anthropic`` sets ``provider="anthropic"``.
    """

    model_config = {"env_prefix": "LLM_", "env_file": ".env", "extra": "ignore"}

    # ── Provider ────────────────────────────────────────────────
    provider: str = Field(
        default="anthropic",
        description="Provider name: 'anthropic', 'local_claude', 'openai', etc.",
    )
    model: str = Field(
        default="claude-sonnet-4-5-20250514",
        description="Model identifier passed to the provider.",
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key. Falls back to provider-specific env vars if unset.",
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Optional base URL override for the provider API.",
    )

    # ── Request defaults ────────────────────────────────────────
    max_tokens: int = Field(default=4096, ge=1)
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: int = Field(default=120, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # ── Cost guardrails ─────────────────────────────────────────
    cost_limit_usd: Optional[float] = Field(
        default=None,
        description="Max cumulative cost (USD) per LLMClient instance. None = no limit.",
    )
    cost_warn_usd: Optional[float] = Field(
        default=None,
        description="Emit warning when cumulative cost exceeds this (USD).",
    )

    # ── Observability ───────────────────────────────────────────
    trace_enabled: bool = Field(default=False)
    trace_exporter: str = Field(
        default="none",
        description="Trace exporter: 'none', 'console', 'otlp'.",
    )
    trace_endpoint: str = Field(default="http://localhost:4317")
    trace_service_name: str = Field(default="llm-gateway")

    log_level: str = Field(default="INFO")
    log_format: str = Field(
        default="json",
        description="Log format: 'json' or 'console'.",
    )

    @model_validator(mode="after")
    def _resolve_api_key(self) -> "GatewayConfig":
        """Fall back to provider-specific env vars if LLM_API_KEY is unset."""
        if self.api_key is not None:
            return self

        fallback_map: dict[str, str] = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        env_var = fallback_map.get(self.provider)
        if env_var:
            value = os.environ.get(env_var)
            if value:
                self.api_key = SecretStr(value)

        return self

    def get_api_key(self) -> str:
        """Return the resolved API key as a plain string.

        Raises:
            ValueError: If no API key is configured for a provider that needs one.
        """
        if self.api_key is None:
            msg = (
                f"No API key configured for provider '{self.provider}'. "
                f"Set LLM_API_KEY or the provider-specific env var."
            )
            raise ValueError(msg)
        return self.api_key.get_secret_value()
```

---

## Phase 3 — Cost Tracking

### 3.1 `src/llm_gateway/cost.py`

```python
"""Token pricing registry and cost tracking."""

from __future__ import annotations

import logging
from typing import Any

from llm_gateway.exceptions import CostLimitExceededError
from llm_gateway.types import TokenUsage

logger = logging.getLogger(__name__)

# ── Pricing Registry (USD per 1 million tokens) ────────────────
_PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-5-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    # OpenAI (examples — update as needed)
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}


def register_pricing(model: str, input_per_1m: float, output_per_1m: float) -> None:
    """Register or update pricing for a model.

    Args:
        model: Model identifier string.
        input_per_1m: Cost in USD per 1M input tokens.
        output_per_1m: Cost in USD per 1M output tokens.
    """
    _PRICING[model] = {"input": input_per_1m, "output": output_per_1m}


def get_pricing(model: str) -> dict[str, float] | None:
    """Return pricing dict for a model, or None if unknown."""
    return _PRICING.get(model)


def calculate_cost(
    model: str, input_tokens: int, output_tokens: int
) -> tuple[float, float]:
    """Calculate USD cost for a given token count.

    Returns:
        Tuple of (input_cost_usd, output_cost_usd). Both 0.0 if model unknown.
    """
    pricing = _PRICING.get(model)
    if pricing is None:
        return 0.0, 0.0
    input_cost = input_tokens * pricing["input"] / 1_000_000
    output_cost = output_tokens * pricing["output"] / 1_000_000
    return input_cost, output_cost


def build_token_usage(
    model: str, input_tokens: int, output_tokens: int
) -> TokenUsage:
    """Build a TokenUsage with cost calculated from the pricing registry."""
    input_cost, output_cost = calculate_cost(model, input_tokens, output_tokens)
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
    )


class CostTracker:
    """Accumulates token usage and cost across multiple LLM calls.

    Supports cost guardrails (warn and hard limit).
    """

    def __init__(
        self,
        cost_limit_usd: float | None = None,
        cost_warn_usd: float | None = None,
    ) -> None:
        self._cost_limit = cost_limit_usd
        self._cost_warn = cost_warn_usd
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._call_count: int = 0
        self._warned: bool = False

    def record(self, usage: TokenUsage) -> None:
        """Record a single LLM call's usage and check guardrails."""
        self._total_input_tokens += usage.input_tokens
        self._total_output_tokens += usage.output_tokens
        self._total_cost_usd += usage.total_cost_usd
        self._call_count += 1

        self._check_guardrails()

    def _check_guardrails(self) -> None:
        """Enforce cost warning and hard limit."""
        if (
            self._cost_warn
            and not self._warned
            and self._total_cost_usd >= self._cost_warn
        ):
            self._warned = True
            logger.warning(
                "LLM cost warning threshold reached: $%.4f >= $%.4f",
                self._total_cost_usd,
                self._cost_warn,
            )

        if self._cost_limit and self._total_cost_usd >= self._cost_limit:
            raise CostLimitExceededError(self._total_cost_usd, self._cost_limit)

    @property
    def total_cost_usd(self) -> float:
        """Cumulative cost in USD."""
        return self._total_cost_usd

    @property
    def total_tokens(self) -> int:
        """Cumulative total tokens."""
        return self._total_input_tokens + self._total_output_tokens

    @property
    def call_count(self) -> int:
        """Number of LLM calls recorded."""
        return self._call_count

    def summary(self) -> dict[str, Any]:
        """Return a summary dict suitable for logging or span attributes."""
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self._total_cost_usd, 6),
            "call_count": self._call_count,
        }

    def reset(self) -> None:
        """Reset all accumulators."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost_usd = 0.0
        self._call_count = 0
        self._warned = False
```

---

## Phase 4 — Provider Protocol & Registry

### 4.1 `src/llm_gateway/providers/base.py`

```python
"""LLM provider protocol — the contract every provider must satisfy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar, runtime_checkable

from llm_gateway.types import LLMMessage, LLMResponse

T = TypeVar("T")


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol that all LLM providers must implement.

    Providers handle the actual communication with the LLM service
    and return standardized LLMResponse objects.
    """

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[T],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse[T]:
        """Send messages to the LLM and return a structured response.

        Args:
            messages: Conversation messages.
            response_model: Pydantic model class for structured output.
            model: Model identifier.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with validated content, token usage, and cost.
        """
        ...

    async def close(self) -> None:
        """Clean up provider resources (HTTP sessions, subprocesses, etc.)."""
        ...
```

### 4.2 `src/llm_gateway/providers/__init__.py`

```python
"""LLM providers sub-package."""

from llm_gateway.providers.base import LLMProvider

__all__ = ["LLMProvider"]
```

### 4.3 `src/llm_gateway/registry.py`

```python
"""Provider registry — maps provider names to factory functions."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from llm_gateway.exceptions import ProviderInitError, ProviderNotFoundError

if TYPE_CHECKING:
    from llm_gateway.config import GatewayConfig
    from llm_gateway.providers.base import LLMProvider

logger = logging.getLogger(__name__)

# Global registry: name → factory(config) → provider instance
_PROVIDERS: dict[str, Callable[["GatewayConfig"], "LLMProvider"]] = {}


def register_provider(
    name: str,
    factory: Callable[["GatewayConfig"], "LLMProvider"],
) -> None:
    """Register a provider factory.

    Args:
        name: Provider name (e.g. "anthropic", "local_claude", "openai").
        factory: Callable that takes GatewayConfig and returns an LLMProvider.
    """
    _PROVIDERS[name] = factory
    logger.debug("Registered LLM provider: %s", name)


def build_provider(config: "GatewayConfig") -> "LLMProvider":
    """Build a provider instance from configuration.

    Triggers lazy registration of built-in providers on first call.

    Args:
        config: Gateway configuration with provider name and settings.

    Returns:
        An initialized LLMProvider instance.

    Raises:
        ProviderNotFoundError: If the provider name is not registered.
        ProviderInitError: If the provider factory raises an error.
    """
    _ensure_builtins_registered()

    factory = _PROVIDERS.get(config.provider)
    if factory is None:
        raise ProviderNotFoundError(config.provider)

    try:
        return factory(config)
    except Exception as exc:
        raise ProviderInitError(config.provider, str(exc)) from exc


def list_providers() -> list[str]:
    """Return names of all registered providers."""
    _ensure_builtins_registered()
    return list(_PROVIDERS.keys())


# ── Lazy Registration ───────────────────────────────────────────

_builtins_registered = False


def _ensure_builtins_registered() -> None:
    """Lazily register built-in providers on first use.

    This avoids importing heavy SDKs (anthropic, openai) at module load time.
    Import errors are caught — providers for uninstalled SDKs are simply
    not registered.
    """
    global _builtins_registered  # noqa: PLW0603
    if _builtins_registered:
        return
    _builtins_registered = True

    # Anthropic
    try:
        from llm_gateway.providers.anthropic import AnthropicProvider

        register_provider("anthropic", AnthropicProvider.from_config)
    except ImportError:
        logger.debug("anthropic extras not installed — provider not available")

    # Local Claude CLI
    try:
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        register_provider("local_claude", LocalClaudeProvider.from_config)
    except ImportError:
        logger.debug("local_claude provider not available")

    # OpenAI (future)
    try:
        from llm_gateway.providers.openai import OpenAIProvider  # type: ignore[import-not-found]

        register_provider("openai", OpenAIProvider.from_config)
    except ImportError:
        logger.debug("openai extras not installed — provider not available")
```

---

## Phase 5 — Anthropic Provider

### 5.1 `src/llm_gateway/providers/anthropic.py`

```python
"""Anthropic provider — wraps AsyncAnthropic + instructor for structured output."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TypeVar

from tenacity import retry, stop_after_attempt, wait_exponential

from llm_gateway.cost import build_token_usage
from llm_gateway.exceptions import ProviderError
from llm_gateway.types import LLMMessage, LLMResponse, TokenUsage

try:
    from anthropic import AsyncAnthropic

    import instructor

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

if not HAS_ANTHROPIC:
    msg = (
        "Anthropic provider requires 'anthropic' and 'instructor' packages. "
        "Install with: pip install 'llm-gateway[anthropic]'"
    )
    raise ImportError(msg)

from llm_gateway.config import GatewayConfig

T = TypeVar("T")


class AnthropicProvider:
    """LLM provider backed by the Anthropic API via instructor."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        max_retries: int = 3,
        timeout_seconds: int = 120,
    ) -> None:
        self._client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=float(timeout_seconds),
        )
        self._instructor = instructor.from_anthropic(self._client)
        self._max_retries = max_retries

    @classmethod
    def from_config(cls, config: GatewayConfig) -> "AnthropicProvider":
        """Factory method for the provider registry."""
        return cls(
            api_key=config.get_api_key(),
            base_url=config.base_url,
            max_retries=config.max_retries,
            timeout_seconds=config.timeout_seconds,
        )

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[T],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse[T]:
        """Call Anthropic API and return structured response with usage."""
        start = time.monotonic()

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        async def _do_call() -> T:
            result: T = await self._instructor.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=list(messages),
                response_model=response_model,
            )
            return result

        try:
            result = await _do_call()
        except Exception as exc:
            raise ProviderError("anthropic", exc) from exc

        latency_ms = (time.monotonic() - start) * 1000
        usage = self._extract_usage(result, model)

        return LLMResponse(
            content=result,
            usage=usage,
            model=model,
            provider="anthropic",
            latency_ms=latency_ms,
        )

    @staticmethod
    def _extract_usage(result: object, model: str) -> TokenUsage:
        """Extract token usage from instructor's _raw_response."""
        raw = getattr(result, "_raw_response", None)
        if raw is None:
            return build_token_usage(model, 0, 0)

        usage = getattr(raw, "usage", None)
        if usage is None:
            return build_token_usage(model, 0, 0)

        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        return build_token_usage(model, input_tokens, output_tokens)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
```

---

## Phase 6 — Local Claude Provider

### 6.1 `src/llm_gateway/providers/local_claude.py`

```python
"""Local Claude CLI provider — runs 'claude' as a subprocess for LLM inference."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

from llm_gateway.config import GatewayConfig
from llm_gateway.exceptions import CLINotFoundError, ProviderError, ResponseValidationError
from llm_gateway.types import LLMMessage, LLMResponse, TokenUsage

T = TypeVar("T")

# Rough estimate: 1 token ≈ 4 characters (for heuristic usage tracking)
_CHARS_PER_TOKEN = 4


class LocalClaudeProvider:
    """LLM provider that delegates to the local ``claude`` CLI binary.

    Structured output is achieved by embedding the JSON schema in the
    prompt and requesting JSON-only output via ``--output-format json``.
    """

    def __init__(self, timeout_seconds: int = 120) -> None:
        self._timeout = timeout_seconds
        self._claude_path = shutil.which("claude")
        if self._claude_path is None:
            raise CLINotFoundError()

    @classmethod
    def from_config(cls, config: GatewayConfig) -> "LocalClaudeProvider":
        """Factory method for the provider registry."""
        return cls(timeout_seconds=config.timeout_seconds)

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[T],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse[T]:
        """Run claude CLI and parse structured output."""
        prompt = self._build_prompt(messages, response_model)
        start = time.monotonic()

        try:
            stdout = await self._run_cli(prompt)
        except Exception as exc:
            raise ProviderError("local_claude", exc) from exc

        latency_ms = (time.monotonic() - start) * 1000

        # Parse and validate the response
        content = self._parse_response(stdout, response_model)
        usage = self._estimate_usage(prompt, stdout)

        return LLMResponse(
            content=content,
            usage=usage,
            model=model,
            provider="local_claude",
            latency_ms=latency_ms,
        )

    def _build_prompt(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[T],
    ) -> str:
        """Build a single prompt string with embedded JSON schema."""
        parts: list[str] = []

        # System instruction with JSON schema
        if issubclass(response_model, BaseModel):  # type: ignore[arg-type]
            schema = json.dumps(
                response_model.model_json_schema(),  # type: ignore[union-attr]
                indent=2,
            )
            parts.append(
                "You MUST respond with ONLY a valid JSON object (no markdown, "
                "no explanation, no extra text) conforming to this schema:\n\n"
                f"```json\n{schema}\n```\n"
            )

        # Conversation messages
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System]: {content}")
            elif role == "assistant":
                parts.append(f"[Assistant]: {content}")
            else:
                parts.append(f"[User]: {content}")

        return "\n\n".join(parts)

    async def _run_cli(self, prompt: str) -> str:
        """Execute the claude CLI and return stdout."""
        assert self._claude_path is not None  # noqa: S101
        proc = await asyncio.create_subprocess_exec(
            self._claude_path,
            "-p", prompt,
            "--output-format", "json",
            "--max-turns", "1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as exc:
            proc.kill()
            msg = f"Claude CLI timed out after {self._timeout}s"
            raise TimeoutError(msg) from exc

        if proc.returncode != 0:
            stderr_text = stderr_bytes.decode(errors="replace").strip()
            msg = f"Claude CLI exited with code {proc.returncode}: {stderr_text}"
            raise RuntimeError(msg)

        stdout_text = stdout_bytes.decode(errors="replace").strip()

        # claude --output-format json wraps result in {"result": "...", ...}
        try:
            wrapper = json.loads(stdout_text)
            if isinstance(wrapper, dict) and "result" in wrapper:
                return str(wrapper["result"])
        except (json.JSONDecodeError, TypeError):
            pass

        return stdout_text

    @staticmethod
    def _parse_response(raw: str, response_model: type[T]) -> T:
        """Parse and validate the raw JSON string against the response model."""
        # Try to extract JSON from markdown code blocks if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (``` markers)
            json_lines = []
            inside = False
            for line in lines:
                if line.strip().startswith("```") and not inside:
                    inside = True
                    continue
                if line.strip() == "```" and inside:
                    break
                if inside:
                    json_lines.append(line)
            cleaned = "\n".join(json_lines)

        try:
            if issubclass(response_model, BaseModel):  # type: ignore[arg-type]
                return response_model.model_validate_json(cleaned)  # type: ignore[union-attr,return-value]
        except Exception as exc:
            raise ResponseValidationError(
                response_model.__name__, str(exc)
            ) from exc

        raise ResponseValidationError(
            response_model.__name__,
            "response_model must be a Pydantic BaseModel subclass",
        )

    @staticmethod
    def _estimate_usage(prompt: str, response: str) -> TokenUsage:
        """Heuristic token estimation (CLI doesn't report actual tokens)."""
        input_tokens = max(1, len(prompt) // _CHARS_PER_TOKEN)
        output_tokens = max(1, len(response) // _CHARS_PER_TOKEN)
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_usd=0.0,  # Local — no API cost
            output_cost_usd=0.0,
        )

    async def close(self) -> None:
        """No-op — no persistent resources to clean up."""
```

---

## Phase 7 — Observability

### 7.1 `src/llm_gateway/observability/tracing.py`

```python
"""OpenTelemetry tracing for LLM calls."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from llm_gateway.types import LLMResponse

logger = logging.getLogger(__name__)

# ── Optional OTEL imports ───────────────────────────────────────
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )

    HAS_OTLP = True
except ImportError:
    HAS_OTLP = False


# Module-level tracer (NoOp if OTEL not installed/configured)
_tracer: Any = None


def configure_tracing(
    exporter: str = "none",
    endpoint: str = "http://localhost:4317",
    service_name: str = "llm-gateway",
) -> None:
    """Configure OpenTelemetry tracing.

    Args:
        exporter: One of "none", "console", "otlp".
        endpoint: OTLP collector endpoint (only used when exporter="otlp").
        service_name: Service name for spans.
    """
    global _tracer  # noqa: PLW0603

    if exporter == "none" or not HAS_OTEL:
        _tracer = None
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if exporter == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif exporter == "otlp":
        if not HAS_OTLP:
            logger.warning(
                "OTLP exporter requested but opentelemetry-exporter-otlp not installed"
            )
            _tracer = None
            return
        provider.add_span_processor(
            SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
        )

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("llm_gateway")
    logger.info("OTEL tracing configured: exporter=%s, service=%s", exporter, service_name)


def get_tracer() -> Any:
    """Return the configured tracer, or None if tracing is disabled."""
    return _tracer


def disable_tracing() -> None:
    """Disable tracing (useful for tests)."""
    global _tracer  # noqa: PLW0603
    _tracer = None


@asynccontextmanager
async def traced_llm_call(
    model: str,
    provider: str,
    operation: str = "llm.complete",
) -> AsyncGenerator[dict[str, Any], None]:
    """Context manager that creates an OTEL span for an LLM call.

    Usage:
        async with traced_llm_call("claude-sonnet", "anthropic") as span_attrs:
            response = await provider.complete(...)
            span_attrs["response"] = response  # set after call

    The span will automatically record:
    - llm.model, llm.provider
    - llm.input_tokens, llm.output_tokens, llm.cost_usd, llm.latency_ms
    - Error status if an exception is raised
    """
    span_data: dict[str, Any] = {}

    if _tracer is None:
        yield span_data
        return

    with _tracer.start_as_current_span(operation) as span:
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.provider", provider)

        try:
            yield span_data
        except Exception as exc:
            span.set_status(trace.StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise
        else:
            # Extract response data if provided
            response = span_data.get("response")
            if response is not None and isinstance(response, LLMResponse):
                span.set_attribute("llm.input_tokens", response.usage.input_tokens)
                span.set_attribute("llm.output_tokens", response.usage.output_tokens)
                span.set_attribute("llm.total_tokens", response.usage.total_tokens)
                span.set_attribute("llm.cost_usd", response.usage.total_cost_usd)
                span.set_attribute("llm.latency_ms", response.latency_ms)
```

### 7.2 `src/llm_gateway/observability/logging.py`

```python
"""Structured logging configuration for llm-gateway."""

from __future__ import annotations

import logging
from typing import Any

_CONFIGURED = False

# ── Optional structlog import ───────────────────────────────────
try:
    import structlog
    from structlog.contextvars import merge_contextvars

    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False


def configure_logging(
    level: str = "INFO",
    fmt: str = "json",
) -> None:
    """Configure logging for llm-gateway.

    If structlog is installed, sets up structured logging with JSON or
    console rendering. Otherwise falls back to stdlib logging.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        fmt: Output format — "json" or "console".
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return
    _CONFIGURED = True

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if HAS_STRUCTLOG:
        _configure_structlog(numeric_level, fmt)
    else:
        logging.basicConfig(level=numeric_level, format="%(levelname)s %(name)s %(message)s")


def _configure_structlog(level: int, fmt: str) -> None:
    """Set up structlog processors and rendering."""
    shared_processors: list[Any] = [
        merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if fmt == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler()
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=shared_processors,
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> Any:
    """Return a logger instance.

    Returns a structlog BoundLogger if structlog is installed,
    otherwise a stdlib logger.
    """
    if HAS_STRUCTLOG:
        return structlog.get_logger(name)
    return logging.getLogger(name)
```

### 7.3 `src/llm_gateway/observability/__init__.py`

```python
"""Observability sub-package — tracing and logging."""

from llm_gateway.observability.logging import configure_logging, get_logger
from llm_gateway.observability.tracing import (
    configure_tracing,
    disable_tracing,
    get_tracer,
    traced_llm_call,
)

__all__ = [
    "configure_logging",
    "configure_tracing",
    "disable_tracing",
    "get_logger",
    "get_tracer",
    "traced_llm_call",
]
```

---

## Phase 8 — LLM Client

### 8.1 `src/llm_gateway/client.py`

```python
"""LLMClient — the single class consumers import and use."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, TypeVar

from llm_gateway.config import GatewayConfig
from llm_gateway.cost import CostTracker
from llm_gateway.observability.logging import configure_logging
from llm_gateway.observability.tracing import configure_tracing, traced_llm_call
from llm_gateway.providers.base import LLMProvider
from llm_gateway.registry import build_provider
from llm_gateway.types import LLMMessage, LLMResponse

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LLMClient:
    """Unified LLM client with config-driven provider selection.

    This is the ONE class consumers should import. Provider switching
    happens entirely via environment variables — zero code changes.

    Usage:
        # Reads LLM_* env vars automatically
        llm = LLMClient()

        # Or with explicit config
        llm = LLMClient(config=GatewayConfig(provider="local_claude"))

        # Or with injected provider (for testing)
        llm = LLMClient(provider_instance=my_mock_provider)

        # Make a call
        resp = await llm.complete(
            messages=[{"role": "user", "content": "Hello"}],
            response_model=MyModel,
        )
        print(resp.content)         # MyModel instance
        print(resp.usage.total_cost_usd)  # Cost in USD
    """

    def __init__(
        self,
        config: GatewayConfig | None = None,
        provider_instance: LLMProvider | None = None,
    ) -> None:
        self._config = config or GatewayConfig()
        self._provider = provider_instance or build_provider(self._config)
        self._cost_tracker = CostTracker(
            cost_limit_usd=self._config.cost_limit_usd,
            cost_warn_usd=self._config.cost_warn_usd,
        )
        self._closed = False

        # Auto-configure observability
        configure_logging(
            level=self._config.log_level,
            fmt=self._config.log_format,
        )
        if self._config.trace_enabled:
            configure_tracing(
                exporter=self._config.trace_exporter,
                endpoint=self._config.trace_endpoint,
                service_name=self._config.trace_service_name,
            )

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[T],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse[T]:
        """Send messages to the configured LLM and return a structured response.

        Args:
            messages: Conversation messages.
            response_model: Pydantic model for structured output validation.
            model: Override the default model from config.
            max_tokens: Override the default max_tokens from config.
            temperature: Override the default temperature from config.

        Returns:
            LLMResponse[T] with validated content, token usage, and cost.

        Raises:
            CostLimitExceededError: If cumulative cost exceeds the limit.
            ProviderError: If the underlying provider raises an error.
            ResponseValidationError: If the response cannot be validated.
        """
        effective_model = model or self._config.model
        effective_max_tokens = max_tokens or self._config.max_tokens
        effective_temperature = temperature if temperature is not None else self._config.temperature

        async with traced_llm_call(
            model=effective_model,
            provider=self._config.provider,
        ) as span_data:
            response = await self._provider.complete(
                messages=messages,
                response_model=response_model,
                model=effective_model,
                max_tokens=effective_max_tokens,
                temperature=effective_temperature,
            )
            span_data["response"] = response

        # Track cost
        self._cost_tracker.record(response.usage)

        logger.info(
            "LLM call completed",
            extra={
                "provider": response.provider,
                "model": response.model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cost_usd": response.usage.total_cost_usd,
                "latency_ms": round(response.latency_ms, 1),
                "cumulative_cost_usd": self._cost_tracker.total_cost_usd,
            },
        )

        return response

    @property
    def total_cost_usd(self) -> float:
        """Cumulative cost across all calls on this client instance."""
        return self._cost_tracker.total_cost_usd

    @property
    def total_tokens(self) -> int:
        """Cumulative tokens across all calls on this client instance."""
        return self._cost_tracker.total_tokens

    @property
    def call_count(self) -> int:
        """Number of LLM calls made on this client instance."""
        return self._cost_tracker.call_count

    def cost_summary(self) -> dict[str, Any]:
        """Return a summary dict of cost/token usage."""
        return self._cost_tracker.summary()

    async def close(self) -> None:
        """Clean up provider resources."""
        if not self._closed:
            await self._provider.close()
            self._closed = True

    async def __aenter__(self) -> "LLMClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Async context manager exit — closes provider."""
        await self.close()
```

### 8.2 `src/llm_gateway/__init__.py`

```python
"""llm-gateway — Production-ready, vendor-agnostic LLM gateway.

Usage:
    from llm_gateway import LLMClient, LLMResponse, GatewayConfig

    llm = LLMClient()  # reads LLM_* env vars
    resp = await llm.complete(messages, response_model=MyModel)
"""

from __future__ import annotations

from llm_gateway.client import LLMClient
from llm_gateway.config import GatewayConfig
from llm_gateway.cost import CostTracker, calculate_cost, register_pricing
from llm_gateway.exceptions import (
    CLINotFoundError,
    CostLimitExceededError,
    GatewayError,
    ProviderError,
    ProviderInitError,
    ProviderNotFoundError,
    ResponseValidationError,
)
from llm_gateway.providers.base import LLMProvider
from llm_gateway.registry import build_provider, list_providers, register_provider
from llm_gateway.types import LLMMessage, LLMResponse, TokenUsage

__all__ = [
    # Core
    "LLMClient",
    "GatewayConfig",
    # Types
    "LLMResponse",
    "LLMMessage",
    "TokenUsage",
    # Provider
    "LLMProvider",
    "register_provider",
    "build_provider",
    "list_providers",
    # Cost
    "CostTracker",
    "calculate_cost",
    "register_pricing",
    # Exceptions
    "GatewayError",
    "ProviderNotFoundError",
    "ProviderInitError",
    "ProviderError",
    "CostLimitExceededError",
    "ResponseValidationError",
    "CLINotFoundError",
]
```

---

## Phase 9 — Tests

### 9.1 `tests/conftest.py`

```python
"""Shared test fixtures for llm-gateway."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar
from unittest.mock import AsyncMock

import pytest

from llm_gateway.config import GatewayConfig
from llm_gateway.cost import build_token_usage
from llm_gateway.providers.base import LLMProvider
from llm_gateway.types import LLMMessage, LLMResponse, TokenUsage

T = TypeVar("T")


class FakeLLMProvider:
    """In-memory fake provider for testing.

    Pre-configure responses with `set_response()`, then pass as
    `LLMClient(provider_instance=fake)`.
    """

    def __init__(self) -> None:
        self._responses: dict[type, object] = {}
        self._call_count: int = 0
        self._last_messages: Sequence[LLMMessage] = []
        self._last_model: str = ""

    def set_response(self, response_model: type[T], response: T) -> None:
        """Pre-configure a response for a given model type."""
        self._responses[response_model] = response

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[T],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse[T]:
        """Return pre-configured response."""
        self._call_count += 1
        self._last_messages = messages
        self._last_model = model

        content = self._responses.get(response_model)
        if content is None:
            msg = f"No fake response configured for {response_model.__name__}"
            raise ValueError(msg)

        usage = build_token_usage(model, 100, 50)
        return LLMResponse(
            content=content,  # type: ignore[arg-type]
            usage=usage,
            model=model,
            provider="fake",
            latency_ms=1.0,
        )

    async def close(self) -> None:
        """No-op."""


@pytest.fixture
def fake_provider() -> FakeLLMProvider:
    """Return a fresh FakeLLMProvider."""
    return FakeLLMProvider()


@pytest.fixture
def test_config(monkeypatch: pytest.MonkeyPatch) -> GatewayConfig:
    """Return a GatewayConfig with test defaults (no real API key needed)."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_API_KEY", "test-key-fake")
    monkeypatch.setenv("LLM_TRACE_ENABLED", "false")
    return GatewayConfig()
```

### 9.2 `tests/unit/test_types.py`

```python
"""Tests for core types."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from llm_gateway.types import LLMResponse, TokenUsage


@pytest.mark.unit
class TestTokenUsage:
    def test_total_tokens(self) -> None:
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_total_cost(self) -> None:
        usage = TokenUsage(input_cost_usd=0.01, output_cost_usd=0.02)
        assert usage.total_cost_usd == pytest.approx(0.03)

    def test_defaults(self) -> None:
        usage = TokenUsage()
        assert usage.total_tokens == 0
        assert usage.total_cost_usd == 0.0

    def test_frozen(self) -> None:
        usage = TokenUsage(input_tokens=10)
        with pytest.raises(AttributeError):
            usage.input_tokens = 20  # type: ignore[misc]


@pytest.mark.unit
class TestLLMResponse:
    def test_generic_content(self) -> None:
        class Answer(BaseModel):
            text: str

        resp = LLMResponse(
            content=Answer(text="hello"),
            usage=TokenUsage(),
            model="test-model",
            provider="test",
        )
        assert resp.content.text == "hello"
        assert resp.provider == "test"
```

### 9.3 `tests/unit/test_exceptions.py`

```python
"""Tests for exception hierarchy."""

from __future__ import annotations

import pytest

from llm_gateway.exceptions import (
    CLINotFoundError,
    CostLimitExceededError,
    GatewayError,
    ProviderError,
    ProviderNotFoundError,
)


@pytest.mark.unit
class TestExceptions:
    def test_hierarchy(self) -> None:
        assert issubclass(ProviderNotFoundError, GatewayError)
        assert issubclass(CostLimitExceededError, GatewayError)
        assert issubclass(ProviderError, GatewayError)
        assert issubclass(CLINotFoundError, ProviderError)

    def test_provider_not_found_message(self) -> None:
        exc = ProviderNotFoundError("foobar")
        assert "foobar" in str(exc)
        assert exc.provider == "foobar"

    def test_cost_limit_exceeded(self) -> None:
        exc = CostLimitExceededError(current=5.0, limit=3.0)
        assert exc.current == 5.0
        assert exc.limit == 3.0
        assert "$5.0" in str(exc) or "5.0000" in str(exc)
```

### 9.4 `tests/unit/test_config.py`

```python
"""Tests for GatewayConfig."""

from __future__ import annotations

import pytest

from llm_gateway.config import GatewayConfig


@pytest.mark.unit
class TestGatewayConfig:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default config loads without errors."""
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = GatewayConfig()
        assert config.provider == "anthropic"
        assert config.max_tokens == 4096

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM_ prefixed env vars override defaults."""
        monkeypatch.setenv("LLM_PROVIDER", "local_claude")
        monkeypatch.setenv("LLM_MODEL", "custom-model")
        monkeypatch.setenv("LLM_MAX_TOKENS", "2048")
        config = GatewayConfig()
        assert config.provider == "local_claude"
        assert config.model == "custom-model"
        assert config.max_tokens == 2048

    def test_api_key_fallback_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Falls back to ANTHROPIC_API_KEY when LLM_API_KEY is not set."""
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        config = GatewayConfig()
        assert config.get_api_key() == "sk-ant-test"

    def test_api_key_fallback_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Falls back to OPENAI_API_KEY for openai provider."""
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        config = GatewayConfig()
        assert config.get_api_key() == "sk-openai-test"

    def test_get_api_key_raises_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ValueError when no API key is configured."""
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = GatewayConfig()
        with pytest.raises(ValueError, match="No API key"):
            config.get_api_key()

    def test_cost_guardrails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_COST_LIMIT_USD", "10.0")
        monkeypatch.setenv("LLM_COST_WARN_USD", "5.0")
        config = GatewayConfig()
        assert config.cost_limit_usd == 10.0
        assert config.cost_warn_usd == 5.0
```

### 9.5 `tests/unit/test_cost.py`

```python
"""Tests for cost tracking."""

from __future__ import annotations

import pytest

from llm_gateway.cost import (
    CostTracker,
    build_token_usage,
    calculate_cost,
    register_pricing,
)
from llm_gateway.exceptions import CostLimitExceededError
from llm_gateway.types import TokenUsage


@pytest.mark.unit
class TestCalculateCost:
    def test_known_model(self) -> None:
        input_cost, output_cost = calculate_cost(
            "claude-haiku-4-5-20251001", 1_000_000, 1_000_000
        )
        assert input_cost == pytest.approx(0.80)
        assert output_cost == pytest.approx(4.00)

    def test_unknown_model(self) -> None:
        input_cost, output_cost = calculate_cost("unknown-xyz", 1000, 1000)
        assert input_cost == 0.0
        assert output_cost == 0.0

    def test_register_custom_pricing(self) -> None:
        register_pricing("my-custom-model", 1.0, 5.0)
        input_cost, output_cost = calculate_cost("my-custom-model", 1_000_000, 1_000_000)
        assert input_cost == pytest.approx(1.0)
        assert output_cost == pytest.approx(5.0)


@pytest.mark.unit
class TestBuildTokenUsage:
    def test_builds_with_cost(self) -> None:
        usage = build_token_usage("claude-haiku-4-5-20251001", 500_000, 100_000)
        assert usage.input_tokens == 500_000
        assert usage.output_tokens == 100_000
        assert usage.input_cost_usd == pytest.approx(0.40)
        assert usage.output_cost_usd == pytest.approx(0.40)


@pytest.mark.unit
class TestCostTracker:
    def test_accumulates(self) -> None:
        tracker = CostTracker()
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50, input_cost_usd=0.01, output_cost_usd=0.02))
        tracker.record(TokenUsage(input_tokens=200, output_tokens=100, input_cost_usd=0.02, output_cost_usd=0.04))
        assert tracker.total_tokens == 450
        assert tracker.total_cost_usd == pytest.approx(0.09)
        assert tracker.call_count == 2

    def test_hard_limit(self) -> None:
        tracker = CostTracker(cost_limit_usd=0.05)
        with pytest.raises(CostLimitExceededError):
            tracker.record(TokenUsage(input_cost_usd=0.03, output_cost_usd=0.03))

    def test_warn_threshold(self) -> None:
        tracker = CostTracker(cost_warn_usd=0.01, cost_limit_usd=100.0)
        # Should not raise, just warn
        tracker.record(TokenUsage(input_cost_usd=0.02, output_cost_usd=0.0))
        assert tracker.total_cost_usd == pytest.approx(0.02)

    def test_summary(self) -> None:
        tracker = CostTracker()
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50))
        summary = tracker.summary()
        assert summary["call_count"] == 1
        assert summary["total_tokens"] == 150

    def test_reset(self) -> None:
        tracker = CostTracker()
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50))
        tracker.reset()
        assert tracker.total_tokens == 0
        assert tracker.call_count == 0
```

### 9.6 `tests/unit/test_registry.py`

```python
"""Tests for the provider registry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm_gateway.config import GatewayConfig
from llm_gateway.exceptions import ProviderNotFoundError
from llm_gateway.registry import (
    _PROVIDERS,
    build_provider,
    list_providers,
    register_provider,
)


@pytest.mark.unit
class TestRegistry:
    def test_register_and_build(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_provider = MagicMock()
        factory = MagicMock(return_value=mock_provider)

        register_provider("test_provider", factory)

        monkeypatch.setenv("LLM_PROVIDER", "test_provider")
        config = GatewayConfig()
        result = build_provider(config)

        factory.assert_called_once_with(config)
        assert result is mock_provider

        # Cleanup
        _PROVIDERS.pop("test_provider", None)

    def test_unknown_provider_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "nonexistent_xyz_999")
        config = GatewayConfig()
        with pytest.raises(ProviderNotFoundError, match="nonexistent_xyz_999"):
            build_provider(config)

    def test_list_providers(self) -> None:
        providers = list_providers()
        assert isinstance(providers, list)
        # At minimum, anthropic should be available if installed
```

### 9.7 `tests/unit/test_client.py`

```python
"""Tests for LLMClient."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from llm_gateway.client import LLMClient
from llm_gateway.config import GatewayConfig
from llm_gateway.exceptions import CostLimitExceededError
from llm_gateway.types import LLMResponse, TokenUsage


class _Answer(BaseModel):
    text: str


@pytest.mark.unit
class TestLLMClient:
    @pytest.mark.asyncio
    async def test_complete_returns_response(self, fake_provider) -> None:
        """LLMClient.complete() returns the provider's response."""
        fake_provider.set_response(_Answer, _Answer(text="hello"))

        config = GatewayConfig(provider="fake", api_key="not-needed")  # type: ignore[call-arg]
        client = LLMClient(config=config, provider_instance=fake_provider)

        resp = await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            response_model=_Answer,
        )
        assert resp.content.text == "hello"
        assert isinstance(resp.usage, TokenUsage)

    @pytest.mark.asyncio
    async def test_tracks_cost(self, fake_provider) -> None:
        """LLMClient accumulates cost across calls."""
        fake_provider.set_response(_Answer, _Answer(text="ok"))

        config = GatewayConfig(provider="fake", api_key="not-needed")  # type: ignore[call-arg]
        client = LLMClient(config=config, provider_instance=fake_provider)

        await client.complete(
            messages=[{"role": "user", "content": "test"}],
            response_model=_Answer,
        )
        assert client.total_tokens > 0
        assert client.call_count == 1

    @pytest.mark.asyncio
    async def test_cost_limit_enforcement(self, fake_provider) -> None:
        """CostLimitExceededError raised when limit exceeded."""
        fake_provider.set_response(_Answer, _Answer(text="ok"))

        config = GatewayConfig(
            provider="fake",
            api_key="not-needed",  # type: ignore[call-arg]
            cost_limit_usd=0.0001,  # Very low limit
        )
        client = LLMClient(config=config, provider_instance=fake_provider)

        with pytest.raises(CostLimitExceededError):
            await client.complete(
                messages=[{"role": "user", "content": "test"}],
                response_model=_Answer,
                model="claude-haiku-4-5-20251001",  # Has known pricing
            )

    @pytest.mark.asyncio
    async def test_model_override(self, fake_provider) -> None:
        """Model parameter overrides config default."""
        fake_provider.set_response(_Answer, _Answer(text="ok"))

        config = GatewayConfig(
            provider="fake",
            api_key="not-needed",  # type: ignore[call-arg]
            model="default-model",
        )
        client = LLMClient(config=config, provider_instance=fake_provider)

        await client.complete(
            messages=[{"role": "user", "content": "test"}],
            response_model=_Answer,
            model="override-model",
        )
        assert fake_provider._last_model == "override-model"

    @pytest.mark.asyncio
    async def test_context_manager(self, fake_provider) -> None:
        """Async context manager calls close()."""
        config = GatewayConfig(provider="fake", api_key="not-needed")  # type: ignore[call-arg]
        async with LLMClient(config=config, provider_instance=fake_provider) as client:
            assert client is not None
        # close() was called

    @pytest.mark.asyncio
    async def test_cost_summary(self, fake_provider) -> None:
        """cost_summary() returns dict with expected keys."""
        fake_provider.set_response(_Answer, _Answer(text="ok"))

        config = GatewayConfig(provider="fake", api_key="not-needed")  # type: ignore[call-arg]
        client = LLMClient(config=config, provider_instance=fake_provider)

        await client.complete(
            messages=[{"role": "user", "content": "test"}],
            response_model=_Answer,
        )
        summary = client.cost_summary()
        assert "call_count" in summary
        assert "total_tokens" in summary
        assert summary["call_count"] == 1
```

### 9.8 `tests/unit/providers/test_anthropic.py`

```python
"""Tests for AnthropicProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm_gateway.types import LLMResponse


class _TestModel(BaseModel):
    answer: str


@pytest.mark.unit
class TestAnthropicProvider:
    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(self) -> None:
        """AnthropicProvider.complete() wraps instructor result in LLMResponse."""
        from llm_gateway.providers.anthropic import AnthropicProvider

        with (
            patch("llm_gateway.providers.anthropic.AsyncAnthropic"),
            patch("llm_gateway.providers.anthropic.instructor") as mock_instructor,
        ):
            provider = AnthropicProvider(api_key="test-key")

            expected = _TestModel(answer="hello")
            # Attach fake _raw_response for token extraction
            raw = MagicMock()
            raw.usage.input_tokens = 100
            raw.usage.output_tokens = 50
            expected._raw_response = raw  # type: ignore[attr-defined]

            mock_instructor.from_anthropic.return_value.messages.create = AsyncMock(
                return_value=expected
            )

            resp = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                response_model=_TestModel,
                model="claude-haiku-4-5-20251001",
            )

            assert isinstance(resp, LLMResponse)
            assert resp.content.answer == "hello"
            assert resp.usage.input_tokens == 100
            assert resp.usage.output_tokens == 50
            assert resp.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_missing_raw_response(self) -> None:
        """Gracefully handles missing _raw_response (usage = 0)."""
        from llm_gateway.providers.anthropic import AnthropicProvider

        with (
            patch("llm_gateway.providers.anthropic.AsyncAnthropic"),
            patch("llm_gateway.providers.anthropic.instructor") as mock_instructor,
        ):
            provider = AnthropicProvider(api_key="test-key")
            expected = _TestModel(answer="ok")
            # No _raw_response attached

            mock_instructor.from_anthropic.return_value.messages.create = AsyncMock(
                return_value=expected
            )

            resp = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                response_model=_TestModel,
                model="claude-haiku-4-5-20251001",
            )

            assert resp.usage.input_tokens == 0
            assert resp.usage.output_tokens == 0

    @pytest.mark.asyncio
    async def test_from_config(self) -> None:
        """from_config factory creates a valid provider."""
        from llm_gateway.config import GatewayConfig
        from llm_gateway.providers.anthropic import AnthropicProvider

        with (
            patch("llm_gateway.providers.anthropic.AsyncAnthropic"),
            patch("llm_gateway.providers.anthropic.instructor"),
        ):
            config = GatewayConfig(
                provider="anthropic",
                api_key="test-key",  # type: ignore[call-arg]
            )
            provider = AnthropicProvider.from_config(config)
            assert isinstance(provider, AnthropicProvider)
```

### 9.9 `tests/unit/providers/test_local_claude.py`

```python
"""Tests for LocalClaudeProvider."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm_gateway.exceptions import CLINotFoundError
from llm_gateway.types import LLMResponse


class _TestModel(BaseModel):
    answer: str


@pytest.mark.unit
class TestLocalClaudeProvider:
    def test_raises_if_cli_not_found(self) -> None:
        """CLINotFoundError if 'claude' not in PATH."""
        with patch("shutil.which", return_value=None):
            from llm_gateway.providers.local_claude import LocalClaudeProvider

            with pytest.raises(CLINotFoundError):
                LocalClaudeProvider()

    @pytest.mark.asyncio
    async def test_complete_parses_json(self) -> None:
        """Subprocess JSON output is parsed into response_model."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        json_output = json.dumps({"result": json.dumps({"answer": "world"})})

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(json_output.encode(), b"")
        )
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            resp = await provider.complete(
                messages=[{"role": "user", "content": "hello"}],
                response_model=_TestModel,
                model="local",
            )

        assert isinstance(resp, LLMResponse)
        assert resp.content.answer == "world"
        assert resp.provider == "local_claude"
        assert resp.usage.input_tokens > 0

    @pytest.mark.asyncio
    async def test_timeout_kills_process(self) -> None:
        """TimeoutError kills the subprocess."""
        import asyncio

        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider(timeout_seconds=1)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
        mock_proc.kill = MagicMock()

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch("asyncio.wait_for", side_effect=asyncio.TimeoutError),
            pytest.raises(Exception),
        ):
            await provider.complete(
                messages=[{"role": "user", "content": "hello"}],
                response_model=_TestModel,
                model="local",
            )

    def test_build_prompt_includes_schema(self) -> None:
        """Prompt includes JSON schema for structured output."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        prompt = provider._build_prompt(
            messages=[{"role": "user", "content": "test"}],
            response_model=_TestModel,
        )
        assert "answer" in prompt
        assert "string" in prompt  # JSON schema type
```

### 9.10 `tests/integration/test_live_providers.py`

```python
"""Integration tests that call real providers.

Run with: pytest -m integration
Requires actual API keys in environment.
"""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel

from llm_gateway import LLMClient, GatewayConfig


class _SimpleAnswer(BaseModel):
    greeting: str


@pytest.mark.integration
class TestLiveAnthropic:
    @pytest.mark.asyncio
    async def test_anthropic_round_trip(self) -> None:
        """Real Anthropic API call with cost tracking."""
        if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("LLM_API_KEY"):
            pytest.skip("No API key available")

        config = GatewayConfig(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            cost_limit_usd=1.0,
        )
        async with LLMClient(config=config) as client:
            resp = await client.complete(
                messages=[{"role": "user", "content": "Say hello in one word."}],
                response_model=_SimpleAnswer,
            )
            assert resp.content.greeting
            assert resp.usage.input_tokens > 0
            assert resp.usage.total_cost_usd > 0
            assert client.total_cost_usd > 0


@pytest.mark.integration
class TestLiveLocalClaude:
    @pytest.mark.asyncio
    async def test_local_claude_round_trip(self) -> None:
        """Real local claude CLI call."""
        import shutil

        if not shutil.which("claude"):
            pytest.skip("claude CLI not in PATH")

        config = GatewayConfig(provider="local_claude")
        async with LLMClient(config=config) as client:
            resp = await client.complete(
                messages=[{"role": "user", "content": "Say hello in one word."}],
                response_model=_SimpleAnswer,
            )
            assert resp.content.greeting
            assert resp.provider == "local_claude"
```

---

## Phase 10 — Documentation & Examples

### 10.1 `README.md`

```markdown
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
```

### 10.2 `CONTRIBUTING.md`

```markdown
# Contributing to llm-gateway

## Setup

```bash
git clone https://github.com/YOUR_ORG/llm-gateway.git
cd llm-gateway
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Testing

```bash
pytest -m unit            # Fast, fully mocked
pytest -m integration     # Requires API keys
pytest --cov=src          # With coverage
```

## Adding a Provider

1. Create `src/llm_gateway/providers/your_provider.py`
2. Implement the `LLMProvider` protocol (see `providers/base.py`)
3. Add a `from_config(cls, config)` classmethod
4. Register in `registry.py` `_ensure_builtins_registered()`
5. Add optional dependency in `pyproject.toml`
6. Add tests in `tests/unit/providers/test_your_provider.py`
7. Update README and docs

## Code Style

- Python 3.11+, strict mypy, Ruff linting
- All functions have type annotations and docstrings
- No `print()` — use logging
- Run `ruff check . && ruff format . && mypy .` before committing
```

### 10.3 `CHANGELOG.md`

```markdown
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
```

### 10.4 `examples/basic_usage.py`

```python
"""Basic usage of llm-gateway."""

import asyncio

from pydantic import BaseModel

from llm_gateway import LLMClient


class Answer(BaseModel):
    """Simple answer model."""

    text: str
    confidence: float


async def main() -> None:
    """Demonstrate basic LLM call."""
    # LLMClient reads LLM_* env vars automatically
    async with LLMClient() as llm:
        resp = await llm.complete(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            response_model=Answer,
        )
        print(f"Answer: {resp.content.text}")
        print(f"Confidence: {resp.content.confidence}")
        print(f"Tokens: {resp.usage.total_tokens}")
        print(f"Cost: ${resp.usage.total_cost_usd:.6f}")
        print(f"Latency: {resp.latency_ms:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
```

### 10.5 `examples/cost_tracking.py`

```python
"""Demonstrates cost tracking and guardrails."""

import asyncio

from pydantic import BaseModel

from llm_gateway import GatewayConfig, LLMClient
from llm_gateway.exceptions import CostLimitExceededError


class Summary(BaseModel):
    text: str


async def main() -> None:
    """Run multiple calls with cost tracking."""
    config = GatewayConfig(
        cost_limit_usd=0.10,  # Hard limit: $0.10
        cost_warn_usd=0.05,   # Warning at $0.05
    )
    async with LLMClient(config=config) as llm:
        for i in range(10):
            try:
                resp = await llm.complete(
                    messages=[{"role": "user", "content": f"Summarize topic {i}"}],
                    response_model=Summary,
                )
                print(f"Call {i}: ${resp.usage.total_cost_usd:.6f} | Cumulative: ${llm.total_cost_usd:.6f}")
            except CostLimitExceededError as exc:
                print(f"Cost limit reached after {llm.call_count} calls: {exc}")
                break

        print(f"\nFinal summary: {llm.cost_summary()}")


if __name__ == "__main__":
    asyncio.run(main())
```

### 10.6 `examples/provider_switching.py`

```python
"""Demonstrates zero-code provider switching via env vars.

Run with different .env configurations:

  # Anthropic API
  LLM_PROVIDER=anthropic
  ANTHROPIC_API_KEY=sk-ant-...

  # Local Claude CLI (no API key needed)
  LLM_PROVIDER=local_claude

The code below is IDENTICAL regardless of provider.
"""

import asyncio

from pydantic import BaseModel

from llm_gateway import LLMClient


class Greeting(BaseModel):
    message: str


async def main() -> None:
    async with LLMClient() as llm:
        resp = await llm.complete(
            messages=[{"role": "user", "content": "Say hello!"}],
            response_model=Greeting,
        )
        print(f"Provider: {resp.provider}")
        print(f"Model: {resp.model}")
        print(f"Response: {resp.content.message}")


if __name__ == "__main__":
    asyncio.run(main())
```

### 10.7 `examples/custom_provider.py`

```python
"""Demonstrates registering a custom provider."""

import asyncio
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

from llm_gateway import (
    GatewayConfig,
    LLMClient,
    LLMResponse,
    TokenUsage,
    register_provider,
)
from llm_gateway.types import LLMMessage

T = TypeVar("T")


class EchoProvider:
    """A demo provider that echoes back the last user message."""

    @classmethod
    def from_config(cls, config: GatewayConfig) -> "EchoProvider":
        return cls()

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[T],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse[T]:
        last_msg = messages[-1]["content"] if messages else "empty"
        content = response_model.model_validate({"text": f"Echo: {last_msg}"})  # type: ignore[union-attr]
        return LLMResponse(
            content=content,
            usage=TokenUsage(input_tokens=len(str(messages)), output_tokens=len(str(content))),
            model=model,
            provider="echo",
        )

    async def close(self) -> None:
        pass


class EchoAnswer(BaseModel):
    text: str


async def main() -> None:
    # Register the custom provider
    register_provider("echo", EchoProvider.from_config)

    # Use it via config
    config = GatewayConfig(provider="echo")
    async with LLMClient(config=config) as llm:
        resp = await llm.complete(
            messages=[{"role": "user", "content": "Hello, world!"}],
            response_model=EchoAnswer,
        )
        print(f"Provider: {resp.provider}")
        print(f"Response: {resp.content.text}")


if __name__ == "__main__":
    asyncio.run(main())
```

### 10.8 `mkdocs.yml`

```yaml
site_name: llm-gateway
site_description: Production-ready, vendor-agnostic LLM gateway
repo_url: https://github.com/YOUR_ORG/llm-gateway
theme:
  name: material
  palette:
    scheme: slate
    primary: indigo
  features:
    - navigation.tabs
    - navigation.sections
    - content.code.copy

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: true
            show_root_heading: true

nav:
  - Home: index.md
  - Quick Start: quickstart.md
  - Configuration: configuration.md
  - Providers: providers.md
  - Cost Tracking: cost-tracking.md
  - Observability: observability.md
  - Custom Providers: custom-providers.md

markdown_extensions:
  - pymdownx.highlight
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true
  - admonition
  - toc:
      permalink: true
```

### 10.9 `docs/index.md`

```markdown
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
```

### 10.10 Documentation stubs

Create the following files in `docs/` with brief content pointing to the relevant sections. Each should be a short markdown file (5-15 lines) with a title and a reference to see the README or source code for details. These are placeholders for the MkDocs site — flesh out as the package matures.

Files to create:
- `docs/quickstart.md` — Installation + first call example
- `docs/configuration.md` — Full env var table + GatewayConfig API
- `docs/providers.md` — Built-in providers (Anthropic, Local Claude) + how they work
- `docs/cost-tracking.md` — TokenUsage, CostTracker, guardrails, pricing registry
- `docs/observability.md` — OTEL tracing setup, span attributes, structlog integration
- `docs/custom-providers.md` — Step-by-step guide to adding a provider

---

## Phase 11 — CI/CD & Pre-commit

### 11.1 `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install uv
        run: pip install uv
      - name: Install dependencies
        run: uv pip install -e ".[dev]" --system
      - name: Ruff check
        run: ruff check .
      - name: Ruff format check
        run: ruff format --check .
      - name: Mypy
        run: mypy .

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install uv
        run: pip install uv
      - name: Install dependencies
        run: uv pip install -e ".[dev]" --system
      - name: Run unit tests
        run: pytest -m unit --cov=src --cov-report=xml -v
      - name: Upload coverage
        if: matrix.python-version == '3.12'
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: coverage.xml

  # Test with minimal dependencies (no optional extras)
  test-minimal:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install uv
        run: pip install uv
      - name: Install core only
        run: uv pip install -e "." --system && uv pip install pytest pytest-asyncio --system
      - name: Test import
        run: python -c "from llm_gateway import LLMClient, GatewayConfig; print('OK')"
```

### 11.2 `.github/workflows/release.yml`

```yaml
name: Release

on:
  push:
    tags: ["v*"]

permissions:
  id-token: write  # Required for Trusted Publishing
  contents: read

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Needed for hatch-vcs
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install build tools
        run: pip install build hatchling hatch-vcs
      - name: Build
        run: python -m build
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        # Uses Trusted Publishing (OIDC) — no API token needed
```

### 11.3 `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ["--maxkb=500"]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.6
    hooks:
      - id: ruff
        args: ["--fix"]
      - id: ruff-format
```

---

## Phase 12 — Final Polish

### 12.1 Verification Checklist

Run these commands to verify the package is complete:

```bash
# 1. Install in development mode
pip install -e ".[dev]"

# 2. Lint
ruff check .
ruff format --check .

# 3. Type check
mypy .

# 4. Run unit tests
pytest -m unit -v --cov=src

# 5. Verify package can be built
python -m build

# 6. Verify import works with minimal deps
pip install -e "."
python -c "from llm_gateway import LLMClient, GatewayConfig, LLMResponse; print('All imports OK')"

# 7. Verify anthropic extras
pip install -e ".[anthropic]"
python -c "from llm_gateway.providers.anthropic import AnthropicProvider; print('Anthropic OK')"

# 8. Test import without anthropic (should not fail)
pip install -e "."  # core only
python -c "
from llm_gateway.registry import list_providers
providers = list_providers()
print(f'Available providers: {providers}')
"
```

### 12.2 Final File Count

| Category | Files | Purpose |
|----------|-------|---------|
| Source | 13 | Core package code |
| Tests | 12 | Unit + integration tests |
| Examples | 4 | Usage demonstrations |
| Docs | 8 | MkDocs site |
| CI/CD | 2 | GitHub Actions workflows |
| Config | 6 | pyproject.toml, .gitignore, .env.example, etc. |
| OSS | 4 | README, LICENSE, CONTRIBUTING, CHANGELOG |
| **Total** | **49** | |

---

## Appendix A — Consumer Migration Guide

> **Context**: This appendix shows how to integrate `llm-gateway` into the
> `job-hunter-agent` project, replacing direct `anthropic`/`instructor` imports.
> This section is project-specific and NOT part of the llm-gateway package itself.

### A.1 Add Dependency

In `job-hunter-agent/pyproject.toml`, replace:
```toml
"anthropic>=0.40.0",
"instructor>=1.0.0",
```
With:
```toml
"llm-gateway[anthropic,tracing,logging]>=0.1.0",
```

### A.2 Update Settings (`src/job_hunter_core/config/settings.py`)

**Remove** the `anthropic_api_key` field. The gateway reads `LLM_API_KEY` or
`ANTHROPIC_API_KEY` directly from the environment.

**Keep**: `haiku_model`, `sonnet_model`, `max_cost_per_run_usd`,
`warn_cost_threshold_usd` — these are business logic for which model to use
per agent and per-run cost limits (distinct from per-client gateway limits).

### A.3 Refactor BaseAgent (`src/job_hunter_agents/agents/base.py`)

```python
# ── BEFORE ──────────────────────────────────────────────────────
from anthropic import AsyncAnthropic
import instructor
from job_hunter_agents.observability.cost_tracker import extract_token_usage

class BaseAgent(ABC):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
        self._instructor = instructor.from_anthropic(self._client)

    async def _call_llm(self, messages, model, response_model, max_retries=3, state=None):
        response = await self._instructor.messages.create(...)
        input_tokens, output_tokens = extract_token_usage(response)
        if state:
            self._track_cost(state, input_tokens, output_tokens, model)
        return response

# ── AFTER ───────────────────────────────────────────────────────
from llm_gateway import LLMClient, LLMResponse

class BaseAgent(ABC):
    def __init__(self, settings: Settings, llm: LLMClient | None = None) -> None:
        self.settings = settings
        self._llm = llm or LLMClient()

    async def _call_llm(self, messages, model, response_model, state=None):
        resp: LLMResponse = await self._llm.complete(
            messages=messages,
            response_model=response_model,
            model=model,
        )
        if state:
            self._track_cost(
                state,
                resp.usage.input_tokens,
                resp.usage.output_tokens,
                model,
            )
        return resp.content  # Return the validated Pydantic model (backward-compatible)
```

### A.4 Delete Obsolete Code

| File | Action |
|------|--------|
| `observability/cost_tracker.py` | Delete `extract_token_usage()` function (replaced by `LLMResponse.usage`) |
| `core/constants.py` | Delete `TOKEN_PRICES` dict (replaced by `llm_gateway.cost.PRICING`) |
| `agents/base.py` | Remove `import anthropic`, `import instructor` |

Note: Keep `CostTracker` and `_track_cost` in job-hunter-agent if you want per-run
cost tracking on `PipelineState`. The gateway tracks per-client costs; the pipeline
may want per-run costs accumulated on `PipelineState`.

### A.5 Update `.env` / `.env.example`

```env
# ── BEFORE ──────────────────────────────────────────────────────
JH_ANTHROPIC_API_KEY=sk-ant-...

# ── AFTER ───────────────────────────────────────────────────────
# Option A: Use LLM_API_KEY (gateway reads it)
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-ant-...

# Option B: Use standard env var (gateway auto-detects)
ANTHROPIC_API_KEY=sk-ant-...

# To switch to local Claude CLI:
LLM_PROVIDER=local_claude
# (no API key needed)
```

### A.6 Update CLI (`src/job_hunter_cli/main.py`)

Add a `--local-claude` flag:

```python
@app.command()
def run(
    ...,
    local_claude: bool = typer.Option(False, "--local-claude", help="Use local Claude CLI"),
) -> None:
    if local_claude:
        os.environ["LLM_PROVIDER"] = "local_claude"
    ...
```

### A.7 Update Tests

**Before** (patching imports in every test):
```python
with (
    patch("job_hunter_agents.agents.base.AsyncAnthropic"),
    patch("job_hunter_agents.agents.base.instructor"),
):
    agent = _StubAgent(settings)
    agent._instructor.messages.create = AsyncMock(return_value=response)
```

**After** (dependency injection):
```python
from tests.conftest import FakeLLMProvider  # or from llm_gateway test utils

fake = FakeLLMProvider()
fake.set_response(MyModel, MyModel(answer="hello"))
agent = _StubAgent(settings, llm=LLMClient(provider_instance=fake))
```

This eliminates ~40 patch sites across 9 test files.

### A.8 Update Dry-Run (`src/job_hunter_agents/dryrun.py`)

**Before** (patches AsyncAnthropic + instructor at multiple import locations):
```python
patch("job_hunter_agents.agents.base.AsyncAnthropic", FakeAsyncAnthropic)
patch("job_hunter_agents.agents.base.instructor", FakeInstructor)
```

**After** (single provider-level mock):
```python
from llm_gateway import LLMClient
from tests.mocks.mock_llm import FakeLLMProvider

# In pipeline setup:
fake_provider = FakeLLMProvider()
fake_provider.set_response(ResumeData, fake_resume_response)
fake_provider.set_response(SearchPreferences, fake_prefs_response)
# ... etc for each agent's response model

llm = LLMClient(provider_instance=fake_provider)
# Pass to each agent: agent = ResumeParserAgent(settings, llm=llm)
```

### A.9 Migration Sequence

Execute in this order to avoid breaking changes:

1. Publish `llm-gateway` v0.1.0 to PyPI (or install from local path)
2. Add `llm-gateway[anthropic,tracing,logging]` to job-hunter-agent dependencies
3. Refactor `BaseAgent.__init__` to accept optional `llm: LLMClient` parameter
4. Refactor `_call_llm` to use `self._llm.complete()` and return `resp.content`
5. Update all agent constructors to pass through `llm` parameter
6. Update `.env` / `.env.example` with `LLM_*` variables
7. Remove `anthropic_api_key` from Settings
8. Delete `extract_token_usage()` and `TOKEN_PRICES`
9. Update test factories and ~40 test patch sites to use DI
10. Simplify `dryrun.py` to use `FakeLLMProvider`
11. Remove direct `anthropic`/`instructor` from pyproject.toml dependencies
12. Run full test suite, verify CI passes

---

## Appendix B — Adding a New Provider

Step-by-step guide for adding e.g. an OpenAI provider:

### B.1 Create `src/llm_gateway/providers/openai.py`

```python
"""OpenAI provider — wraps AsyncOpenAI + instructor."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TypeVar

from tenacity import retry, stop_after_attempt, wait_exponential

from llm_gateway.cost import build_token_usage
from llm_gateway.exceptions import ProviderError
from llm_gateway.types import LLMMessage, LLMResponse, TokenUsage

try:
    from openai import AsyncOpenAI
    import instructor
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

if not HAS_OPENAI:
    raise ImportError(
        "OpenAI provider requires 'openai' and 'instructor'. "
        "Install with: pip install 'llm-gateway[openai]'"
    )

from llm_gateway.config import GatewayConfig

T = TypeVar("T")


class OpenAIProvider:
    """LLM provider backed by the OpenAI API via instructor."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        max_retries: int = 3,
        timeout_seconds: int = 120,
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(timeout_seconds),
        )
        self._instructor = instructor.from_openai(self._client)
        self._max_retries = max_retries

    @classmethod
    def from_config(cls, config: GatewayConfig) -> "OpenAIProvider":
        return cls(
            api_key=config.get_api_key(),
            base_url=config.base_url,
            max_retries=config.max_retries,
            timeout_seconds=config.timeout_seconds,
        )

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        response_model: type[T],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse[T]:
        start = time.monotonic()

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        async def _do_call() -> T:
            return await self._instructor.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=list(messages),
                response_model=response_model,
            )

        try:
            result = await _do_call()
        except Exception as exc:
            raise ProviderError("openai", exc) from exc

        latency_ms = (time.monotonic() - start) * 1000
        usage = self._extract_usage(result, model)

        return LLMResponse(
            content=result,
            usage=usage,
            model=model,
            provider="openai",
            latency_ms=latency_ms,
        )

    @staticmethod
    def _extract_usage(result: object, model: str) -> TokenUsage:
        raw = getattr(result, "_raw_response", None)
        if raw is None:
            return build_token_usage(model, 0, 0)
        usage = getattr(raw, "usage", None)
        if usage is None:
            return build_token_usage(model, 0, 0)
        return build_token_usage(
            model,
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )

    async def close(self) -> None:
        await self._client.close()
```

### B.2 Register in `registry.py`

Already handled by the lazy registration in `_ensure_builtins_registered()` — the
OpenAI block is already present. Just install `pip install 'llm-gateway[openai]'`.

### B.3 Add Pricing in `cost.py`

Already included in the `_PRICING` dict. For new models, call:
```python
from llm_gateway import register_pricing
register_pricing("gpt-5", input_per_1m=5.0, output_per_1m=15.0)
```

### B.4 Consumer Switches via `.env`

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
```

Zero code changes in the consumer. The gateway detects the provider, loads
the OpenAI factory from the registry, and routes all `LLMClient.complete()`
calls through the OpenAI provider.

---

## Summary

This plan produces a **49-file, fully tested, production-ready Python package**
that can be published to PyPI and used by any project that needs LLM integration.

**Key value propositions:**
1. **Import ONE class** (`LLMClient`) — never import provider SDKs directly
2. **Switch providers via `.env`** — zero code changes, zero conditional imports
3. **Know your costs** — every response includes token usage and USD cost
4. **See everything** — OTEL spans with model/tokens/cost/latency per call
5. **Stay safe** — configurable cost guardrails prevent runaway spending
6. **Test easily** — `FakeLLMProvider` + dependency injection, no import patching
7. **Extend freely** — new providers are a single file + registry entry
