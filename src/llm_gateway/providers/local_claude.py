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
    def from_config(cls, config: GatewayConfig) -> LocalClaudeProvider:
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
        if issubclass(response_model, BaseModel):
            schema = json.dumps(
                response_model.model_json_schema(),
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
        assert self._claude_path is not None
        proc = await asyncio.create_subprocess_exec(
            self._claude_path,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--max-turns",
            "1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._timeout,
            )
        except TimeoutError as exc:
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
            if issubclass(response_model, BaseModel):
                return response_model.model_validate_json(cleaned)  # type: ignore[return-value]
        except Exception as exc:
            raise ResponseValidationError(response_model.__name__, str(exc)) from exc

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
