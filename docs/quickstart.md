# Quick Start

## Installation

```bash
pip install 'llm-gateway[anthropic]'
```

## First Call

```python
import asyncio
from pydantic import BaseModel
from llm_gateway import LLMClient

class Answer(BaseModel):
    text: str

async def main():
    async with LLMClient() as llm:
        resp = await llm.complete(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            response_model=Answer,
        )
        print(resp.content.text)
        print(resp.usage.total_cost_usd)

asyncio.run(main())
```

See [Configuration](configuration.md) for all available settings.
