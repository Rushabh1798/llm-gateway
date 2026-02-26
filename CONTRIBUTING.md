# Contributing to llm-gateway

## Setup

```bash
git clone https://github.com/YOUR_ORG/llm-gateway.git
cd llm-gateway
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

For integration tests (optional):

```bash
cd integration_tests
pip install -e .
```

## Testing

```bash
# Unit tests (39 tests, fast, fully mocked)
pytest -m unit -v

# Integration tests — dry-run (22 tests, mocked, no real LLM calls)
cd integration_tests && pytest -v

# Integration tests — live (10 tests, requires `claude` CLI in PATH)
cd integration_tests && pytest --run-live -m live -v

# Unit tests with coverage
pytest -m unit --cov=src --cov-report=term-missing -v
```

## Pre-commit

Pre-commit hooks mirror the CI pipeline. They run automatically on `git commit`:

```bash
pre-commit run --all-files        # manual run
```

Hooks: trailing whitespace, EOF fixer, YAML check, ruff lint + format, mypy, unit tests, integration dry-run tests.

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
- Run `ruff check . && ruff format --check . && mypy .` before committing
