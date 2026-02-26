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
        cost_warn_usd=0.05,  # Warning at $0.05
    )
    async with LLMClient(config=config) as llm:
        for i in range(10):
            try:
                resp = await llm.complete(
                    messages=[{"role": "user", "content": f"Summarize topic {i}"}],
                    response_model=Summary,
                )
                print(
                    f"Call {i}: ${resp.usage.total_cost_usd:.6f} | Cumulative: ${llm.total_cost_usd:.6f}"
                )
            except CostLimitExceededError as exc:
                print(f"Cost limit reached after {llm.call_count} calls: {exc}")
                break

        print(f"\nFinal summary: {llm.cost_summary()}")


if __name__ == "__main__":
    asyncio.run(main())
