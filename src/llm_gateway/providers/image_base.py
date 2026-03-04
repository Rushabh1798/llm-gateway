"""Image generation provider protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from llm_gateway.types import ImageGenerationResponse


@runtime_checkable
class ImageGenerationProvider(Protocol):
    """Protocol that all image generation providers must implement."""

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
            model: Model identifier. ``None`` means provider picks its default.
            width: Image width in pixels.
            height: Image height in pixels.
            num_images: Number of images to generate.
            quality: Quality tier (e.g. "standard", "hd").

        Returns:
            ImageGenerationResponse with generated image data and cost.
        """
        ...

    async def close(self) -> None:
        """Clean up provider resources."""
        ...
