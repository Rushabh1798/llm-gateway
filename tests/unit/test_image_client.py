"""Unit tests for ImageClient."""

from __future__ import annotations

import pytest

from llm_gateway import (
    FakeImageProvider,
    GatewayConfig,
    ImageClient,
    ImageData,
    ImageGenerationResponse,
    ImageTokenUsage,
)


@pytest.mark.unit
class TestImageClient:
    """Tests for ImageClient with FakeImageProvider."""

    @pytest.fixture
    def fake_provider(self) -> FakeImageProvider:
        """Create a FakeImageProvider instance."""
        return FakeImageProvider()

    @pytest.fixture
    def client(self, fake_provider: FakeImageProvider) -> ImageClient:
        """Create an ImageClient with fake provider."""
        config = GatewayConfig(image_provider="fake_image")
        return ImageClient(config=config, provider_instance=fake_provider)

    @pytest.mark.asyncio
    async def test_generate_image_returns_response(
        self, client: ImageClient, fake_provider: FakeImageProvider
    ) -> None:
        """Default fake response has a URL and records the call."""
        resp = await client.generate_image("a cat wearing a hat")
        assert len(resp.images) == 1
        assert resp.images[0].url is not None
        assert "fake-image-provider" in (resp.images[0].url or "")
        assert resp.provider == "fake_image"
        assert fake_provider.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_multiple_images(
        self, client: ImageClient, fake_provider: FakeImageProvider
    ) -> None:
        """Requesting multiple images returns that many."""
        resp = await client.generate_image("dogs", num_images=3)
        assert len(resp.images) == 3
        assert fake_provider.calls[0].num_images == 3

    @pytest.mark.asyncio
    async def test_custom_response(self, fake_provider: FakeImageProvider) -> None:
        """set_response overrides the default."""
        custom = ImageGenerationResponse(
            images=[ImageData(url="https://custom.test/img.png", revised_prompt="custom cat")],
            usage=ImageTokenUsage(prompt_tokens=10, total_cost_usd=0.05),
            model="custom-model",
            provider="fake_image",
        )
        fake_provider.set_response(custom)

        config = GatewayConfig(image_provider="fake_image")
        client = ImageClient(config=config, provider_instance=fake_provider)

        resp = await client.generate_image("a cat")
        assert resp.images[0].url == "https://custom.test/img.png"
        assert resp.usage.total_cost_usd == 0.05

    @pytest.mark.asyncio
    async def test_cost_tracking(
        self, client: ImageClient, fake_provider: FakeImageProvider
    ) -> None:
        """Cost accumulates across calls."""
        fake_provider_with_cost = FakeImageProvider(default_cost_usd=0.04)
        config = GatewayConfig(image_provider="fake_image")
        c = ImageClient(config=config, provider_instance=fake_provider_with_cost)

        await c.generate_image("cat")
        await c.generate_image("dog")
        assert c.total_cost_usd == pytest.approx(0.08)
        assert c.call_count == 2

    @pytest.mark.asyncio
    async def test_cost_summary(self, client: ImageClient) -> None:
        """cost_summary returns structured dict."""
        await client.generate_image("cat")
        summary = client.cost_summary()
        assert "total_cost_usd" in summary
        assert "call_count" in summary
        assert summary["call_count"] == 1

    @pytest.mark.asyncio
    async def test_context_manager(self, fake_provider: FakeImageProvider) -> None:
        """Async context manager calls close."""
        config = GatewayConfig(image_provider="fake_image")
        async with ImageClient(config=config, provider_instance=fake_provider) as client:
            resp = await client.generate_image("cat")
            assert len(resp.images) == 1

    @pytest.mark.asyncio
    async def test_prompt_recorded(
        self, client: ImageClient, fake_provider: FakeImageProvider
    ) -> None:
        """Prompts are recorded in fake provider calls."""
        await client.generate_image("a sunset over mountains")
        assert fake_provider.calls[0].prompt == "a sunset over mountains"

    @pytest.mark.asyncio
    async def test_model_passthrough(
        self, client: ImageClient, fake_provider: FakeImageProvider
    ) -> None:
        """Model parameter is passed through to provider."""
        await client.generate_image("cat", model="dall-e-3")
        assert fake_provider.calls[0].model == "dall-e-3"

    @pytest.mark.asyncio
    async def test_quality_passthrough(
        self, client: ImageClient, fake_provider: FakeImageProvider
    ) -> None:
        """Quality parameter is passed through to provider."""
        await client.generate_image("cat", quality="hd")
        assert fake_provider.calls[0].quality == "hd"
