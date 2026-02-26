# llm-gateway Integration Tests

Self-contained Python project that validates `llm-gateway` as an **external consumer** would use it — installed via git SHA dependency, not imported from the parent source tree.

## How It Works

- `pyproject.toml` declares `llm-gateway` as a git/file dependency (pinned to a commit SHA)
- Tests import `llm_gateway` as any consumer project would
- **Dry-run mode** (default): all LLM calls are mocked — fast, no API keys needed
- **Live mode** (`--run-live`): calls the real `claude` CLI — validates end-to-end

## Setup

```bash
cd integration_tests
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Running Tests

```bash
# Dry-run only (default) — mocked, no real LLM calls
pytest -v

# Live tests only — requires `claude` CLI in PATH
pytest -v --run-live -m live

# Everything — dry-run + live
pytest -v --run-live
```

## Switching to Remote Git Dependency

Edit `pyproject.toml` and replace the file:// reference:

```toml
dependencies = [
    "llm-gateway @ git+https://github.com/YOUR_ORG/llm-gateway.git@f6bbd7e04e94f0cf1e2e191f1040d82d7ac29063",
    ...
]
```
