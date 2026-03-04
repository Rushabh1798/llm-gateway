"""ImageClient — image generation counterpart to LLMClient."""

from __future__ import annotations

import logging
from typing import Any

from llm_gateway.config import GatewayConfig
from llm_gateway.cost import ImageCostTracker
from llm_gateway.observability.tracing import traced_image_call
from llm_gateway.providers.image_base import ImageGenerationProvider
from llm_gateway.registry import build_image_provider
from llm_gateway.types import ImageGenerationResponse

logger = logging.getLogger(__name__)


class ImageClient:
    """Unified image generation client with config-driven provider selection.

    Mirrors ``LLMClient`` design. Provider switching happens via
    environment variables — zero code changes.

    Usage::

        client = ImageClient()  # reads LLM_IMAGE_PROVIDER env var
        resp = await client.generate_image("a cat wearing a hat")
        print(resp.images[0].url)
        print(client.total_cost_usd)
    """

    def __init__(
        self,
        config: GatewayConfig | None = None,
        provider_instance: ImageGenerationProvider | None = None,
    ) -> None:
        self._config = config or GatewayConfig()
        self._provider = provider_instance or build_image_provider(self._config)
        self._cost_tracker = ImageCostTracker(
            cost_limit_usd=self._config.cost_limit_usd,
            cost_warn_usd=self._config.cost_warn_usd,
        )
        self._closed = False

    async def generate_image(
        self,
        prompt: str,
        model: str | None = None,
        width: int | None = None,
        height: int | None = None,
        num_images: int = 1,
        quality: str = "standard",
    ) -> ImageGenerationResponse:
        """Generate images from a text prompt.

        Args:
            prompt: Text description of the desired image.
            model: Override the default model.
            width: Image width in pixels.
            height: Image height in pixels.
            num_images: Number of images to generate.
            quality: Quality tier (e.g. "standard", "hd").

        Returns:
            ImageGenerationResponse with generated images and cost info.

        Raises:
            CostLimitExceededError: If cumulative cost exceeds the limit.
            ProviderError: If the underlying provider raises an error.
        """
        async with traced_image_call(
            model=model,
            provider=self._config.image_provider,
        ) as span_data:
            response = await self._provider.generate_image(
                prompt=prompt,
                model=model,
                width=width,
                height=height,
                num_images=num_images,
                quality=quality,
            )
            span_data["response"] = response

        self._cost_tracker.record(response.usage)

        logger.info(
            "Image generation completed",
            extra={
                "provider": response.provider,
                "model": response.model,
                "num_images": len(response.images),
                "cost_usd": response.usage.total_cost_usd,
                "latency_ms": round(response.latency_ms, 1),
                "cumulative_cost_usd": self._cost_tracker.total_cost_usd,
            },
        )

        return response

    @property
    def total_cost_usd(self) -> float:
        """Cumulative cost across all calls on this client instance."""
        return self._cost_tracker.total_cost_usd

    @property
    def call_count(self) -> int:
        """Number of image generation calls made."""
        return self._cost_tracker.call_count

    def cost_summary(self) -> dict[str, Any]:
        """Return a summary dict of cost/usage."""
        return self._cost_tracker.summary()

    async def close(self) -> None:
        """Clean up provider resources."""
        if not self._closed:
            await self._provider.close()
            self._closed = True

    async def __aenter__(self) -> ImageClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Async context manager exit — closes provider."""
        await self.close()
