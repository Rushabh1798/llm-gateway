"""Tests for LocalClaudeProvider."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm_gateway.exceptions import CLINotFoundError, ProviderError
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
