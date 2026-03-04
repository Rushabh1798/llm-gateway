"""OpenAI image generation provider (GPT Image / DALL-E)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from tenacity import retry, stop_after_attempt, wait_exponential

from llm_gateway.cost import build_image_usage
from llm_gateway.exceptions import ProviderError
from llm_gateway.types import ImageData, ImageGenerationResponse

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from llm_gateway.config import GatewayConfig


class OpenAIImageProvider:
    """Image generation via OpenAI's API (gpt-image-1, dall-e-3, dall-e-2)."""

    DEFAULT_MODEL = "gpt-image-1"

    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        timeout_seconds: int = 120,
    ) -> None:
        try:
            from openai import AsyncOpenAI as _AsyncOpenAI
        except ImportError as exc:
            msg = "openai package required: pip install 'llm-gateway[openai]'"
            raise ImportError(msg) from exc

        self._client: AsyncOpenAI = _AsyncOpenAI(
            api_key=api_key,
            timeout=float(timeout_seconds),
        )
        self._max_retries = max_retries

    @classmethod
    def from_config(cls, config: GatewayConfig) -> OpenAIImageProvider:
        """Factory for provider registry."""
        return cls(
            api_key=config.get_api_key(),
            max_retries=config.max_retries,
            timeout_seconds=config.timeout_seconds,
        )

    async def generate_image(
        self,
        prompt: str,
        model: str | None = None,
        width: int | None = None,
        height: int | None = None,
        num_images: int = 1,
        quality: str = "standard",
    ) -> ImageGenerationResponse:
        """Generate images via OpenAI API.

        Args:
            prompt: Text description of the desired image.
            model: Model identifier (gpt-image-1, dall-e-3, dall-e-2).
            width: Image width in pixels.
            height: Image height in pixels.
            num_images: Number of images to generate.
            quality: Quality tier (standard, hd, low, medium, high).

        Returns:
            ImageGenerationResponse with generated image data and cost.
        """
        effective_model = model or self.DEFAULT_MODEL
        size = self._resolve_size(width, height)

        start = time.monotonic()

        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "model": effective_model,
            "n": num_images,
            "quality": quality,
        }
        if size is not None:
            kwargs["size"] = size

        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        async def _do_call() -> Any:
            return await self._client.images.generate(**kwargs)

        try:
            result = await _do_call()
        except Exception as exc:
            raise ProviderError("openai_image", exc) from exc

        latency_ms = (time.monotonic() - start) * 1000

        images = [
            ImageData(
                url=getattr(img, "url", None),
                b64_json=getattr(img, "b64_json", None),
                revised_prompt=getattr(img, "revised_prompt", "") or "",
            )
            for img in result.data
        ]

        usage = build_image_usage(
            model=effective_model,
            quality=quality,
            size=size or "auto",
            num_images=len(images),
        )

        return ImageGenerationResponse(
            images=images,
            usage=usage,
            model=effective_model,
            provider="openai_image",
            latency_ms=latency_ms,
        )

    @staticmethod
    def _resolve_size(width: int | None, height: int | None) -> str | None:
        """Convert width/height to OpenAI size string."""
        if width is None and height is None:
            return None
        w = width or 1024
        h = height or 1024
        return f"{w}x{h}"

    async def close(self) -> None:
        """Clean up HTTP client."""
        await self._client.close()
