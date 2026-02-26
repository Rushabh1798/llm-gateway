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
