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
