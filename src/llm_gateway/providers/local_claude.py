"""Local Claude CLI provider — runs 'claude' as a subprocess for LLM inference."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

from llm_gateway.config import GatewayConfig
from llm_gateway.exceptions import CLINotFoundError, ProviderError, ResponseValidationError
from llm_gateway.types import LLMMessage, LLMResponse, TokenUsage

T = TypeVar("T")

logger = logging.getLogger(__name__)

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
        logger.debug(
            "claude_cli_request | model=%s response_model=%s prompt_length=%d\n%s",
            model,
            response_model.__name__,
            len(prompt),
            prompt[:500],
        )
        start = time.monotonic()

        try:
            result_text, wrapper_meta = await self._run_cli(prompt)
        except Exception as exc:
            logger.error("claude_cli_error | %s: %s", type(exc).__name__, exc)
            raise ProviderError("local_claude", exc) from exc

        latency_ms = (time.monotonic() - start) * 1000

        # Parse and validate the response
        content = self._parse_response(result_text, response_model)
        usage = self._build_usage(prompt, result_text, wrapper_meta)

        if isinstance(content, BaseModel):
            fields = content.model_dump()
            reply = " | ".join(f"{k}={v}" for k, v in fields.items())
        else:
            reply = str(content)[:200]
        logger.info(
            "claude_cli_complete | latency=%.0fms tokens=%d+%d cost=$%.4f\n  -> %s",
            latency_ms,
            usage.input_tokens,
            usage.output_tokens,
            usage.total_cost_usd,
            reply,
        )

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

    async def _run_cli(self, prompt: str) -> tuple[str, dict[str, object]]:
        """Execute the claude CLI and return (result_text, wrapper_metadata)."""
        assert self._claude_path is not None

        # Strip CLAUDECODE env var to allow running inside a Claude Code session
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

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
            env=env,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._timeout,
            )
        except TimeoutError as exc:
            proc.kill()
            logger.error("claude_cli_timeout | timeout=%ds", self._timeout)
            msg = f"Claude CLI timed out after {self._timeout}s"
            raise TimeoutError(msg) from exc

        stdout_text = stdout_bytes.decode(errors="replace").strip()
        stderr_text = stderr_bytes.decode(errors="replace").strip()

        if stderr_text:
            logger.warning("claude_cli_stderr | %s", stderr_text[:500])

        if proc.returncode != 0:
            msg = f"Claude CLI exited with code {proc.returncode}: {stderr_text}"
            raise RuntimeError(msg)

        # claude --output-format json wraps result in {"result": "...", ...}
        try:
            wrapper = json.loads(stdout_text)
            if isinstance(wrapper, dict) and "result" in wrapper:
                result_text = str(wrapper["result"])
                logger.debug(
                    "claude_cli_raw_result | duration=%dms cost=$%.6f\n%s",
                    wrapper.get("duration_ms", 0),
                    wrapper.get("total_cost_usd", 0.0),
                    result_text[:2000],
                )
                return result_text, wrapper
        except (json.JSONDecodeError, TypeError):
            pass

        logger.debug(
            "claude_cli_raw_fallback | stdout_length=%d\n%s", len(stdout_text), stdout_text[:1000]
        )
        return stdout_text, {}

    @staticmethod
    def _parse_response(raw: str, response_model: type[T]) -> T:
        """Parse and validate the raw JSON string against the response model."""
        cleaned = raw.strip()

        # Extract JSON from markdown code blocks (```json ... ```)
        # The result text often contains leading newlines before the fence
        if "```" in cleaned:
            json_lines: list[str] = []
            inside = False
            for line in cleaned.split("\n"):
                stripped = line.strip()
                if stripped.startswith("```") and not inside:
                    inside = True
                    continue
                if stripped == "```" and inside:
                    break
                if inside:
                    json_lines.append(line)
            if json_lines:
                cleaned = "\n".join(json_lines)

        # Try direct JSON parsing (no code block wrapper)
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
    def _build_usage(prompt: str, response: str, wrapper: dict[str, object]) -> TokenUsage:
        """Build token usage from CLI wrapper metadata, falling back to heuristics."""
        # Extract real token counts from the wrapper's usage/modelUsage fields
        usage_data = wrapper.get("usage", {})
        model_usage = wrapper.get("modelUsage", {})
        raw_cost = wrapper.get("total_cost_usd")
        total_cost = float(str(raw_cost)) if raw_cost is not None else 0.0

        input_tokens = 0
        output_tokens = 0

        if isinstance(usage_data, dict):
            input_tokens = int(usage_data.get("input_tokens", 0) or 0)
            output_tokens = int(usage_data.get("output_tokens", 0) or 0)
            # Include cache tokens in input count for accurate tracking
            cache_read = int(usage_data.get("cache_read_input_tokens", 0) or 0)
            cache_create = int(usage_data.get("cache_creation_input_tokens", 0) or 0)
            input_tokens += cache_read + cache_create

        # If wrapper had no usage data, fall back to heuristic
        if input_tokens == 0 and output_tokens == 0:
            input_tokens = max(1, len(prompt) // _CHARS_PER_TOKEN)
            output_tokens = max(1, len(response) // _CHARS_PER_TOKEN)

        # Split cost evenly between input/output if we have a total
        input_cost = total_cost * 0.5
        output_cost = total_cost * 0.5

        # Try to get per-model cost breakdown
        if isinstance(model_usage, dict):
            for _model_name, model_data in model_usage.items():
                if isinstance(model_data, dict) and "costUSD" in model_data:
                    total_cost = float(model_data["costUSD"])
                    input_cost = total_cost * 0.5
                    output_cost = total_cost * 0.5
                    break

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
        )

    async def close(self) -> None:
        """No-op — no persistent resources to clean up."""
