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
