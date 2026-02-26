"""Tests for LocalClaudeProvider."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm_gateway.config import GatewayConfig
from llm_gateway.exceptions import CLINotFoundError, ProviderError, ResponseValidationError
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
        mock_proc.communicate = AsyncMock(return_value=(json_output.encode(), b""))
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
            pytest.raises(ProviderError),
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

    def test_build_prompt_system_and_assistant_roles(self) -> None:
        """Prompt formats system and assistant roles correctly."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        prompt = provider._build_prompt(
            messages=[
                {"role": "system", "content": "You are a helpful bot."},
                {"role": "user", "content": "Hi there."},
                {"role": "assistant", "content": "Hello!"},
            ],
            response_model=_TestModel,
        )
        assert "[System]: You are a helpful bot." in prompt
        assert "[User]: Hi there." in prompt
        assert "[Assistant]: Hello!" in prompt

    def test_from_config_factory(self) -> None:
        """from_config creates a provider with timeout from config."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            config = GatewayConfig(provider="local_claude", timeout_seconds=300)
            provider = LocalClaudeProvider.from_config(config)
            assert provider._timeout == 300

    def test_parse_response_valid_json(self) -> None:
        """_parse_response parses valid JSON against response model."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        result = LocalClaudeProvider._parse_response('{"answer": "42"}', _TestModel)
        assert isinstance(result, _TestModel)
        assert result.answer == "42"

    def test_parse_response_markdown_code_block(self) -> None:
        """_parse_response strips markdown code fences before parsing."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        raw = '```json\n{"answer": "hello"}\n```'
        result = LocalClaudeProvider._parse_response(raw, _TestModel)
        assert isinstance(result, _TestModel)
        assert result.answer == "hello"

    def test_parse_response_invalid_json_raises(self) -> None:
        """_parse_response raises ResponseValidationError on invalid JSON."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with pytest.raises(ResponseValidationError, match="_TestModel"):
            LocalClaudeProvider._parse_response("not valid json", _TestModel)

    @pytest.mark.asyncio
    async def test_run_cli_nonzero_exit_raises(self) -> None:
        """Non-zero exit code from CLI raises RuntimeError wrapped in ProviderError."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"some error"))
        mock_proc.returncode = 1

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            pytest.raises(ProviderError, match="local_claude"),
        ):
            await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                response_model=_TestModel,
                model="local",
            )

    @pytest.mark.asyncio
    async def test_run_cli_raw_text_fallback(self) -> None:
        """Raw text output (not JSON wrapper) is used as-is for parsing."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()

        # Return raw JSON directly (no {"result": ...} wrapper)
        raw_json = '{"answer": "direct"}'
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(raw_json.encode(), b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            resp = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                response_model=_TestModel,
                model="local",
            )
        assert resp.content.answer == "direct"

    @pytest.mark.asyncio
    async def test_close_is_noop(self) -> None:
        """close() completes without error."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        with patch("shutil.which", return_value="/usr/bin/claude"):
            provider = LocalClaudeProvider()
        await provider.close()  # Should not raise

    def test_build_usage_heuristic_fallback(self) -> None:
        """_build_usage falls back to heuristic when wrapper has no usage data."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        usage = LocalClaudeProvider._build_usage("hello world", '{"answer": "ok"}', {})
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        assert usage.input_cost_usd == 0.0
        assert usage.output_cost_usd == 0.0

    def test_build_usage_from_wrapper(self) -> None:
        """_build_usage extracts real token counts from wrapper metadata."""
        from llm_gateway.providers.local_claude import LocalClaudeProvider

        wrapper: dict[str, object] = {
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "total_cost_usd": 0.01,
        }
        usage = LocalClaudeProvider._build_usage("prompt", "response", wrapper)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_cost_usd == pytest.approx(0.01, abs=0.001)
