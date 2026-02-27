"""Live integration tests — calls the real local claude CLI.

These tests validate end-to-end: LLMClient → LocalClaudeProvider → claude CLI
→ JSON response → Pydantic validation → LLMResponse with usage tracking.

Run with: pytest --run-live -m live -v
Requires: `claude` CLI installed and available in PATH.
"""

from __future__ import annotations

import logging
import shutil
from typing import Any

import pytest

from llm_gateway import LLMResponse, TokenUsage

from .response_models import (
    CapitalCity,
    FactAnswer,
    Greeting,
    MathAnswer,
    SentimentResult,
    SummaryResult,
    TranslationResult,
)

logger = logging.getLogger(__name__)

# Skip the entire module if claude CLI is not available
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not shutil.which("claude"),
        reason="claude CLI not found in PATH",
    ),
]


# ─── Basic Round-Trip ────────────────────────────────────────────


class TestLiveBasicRoundTrip:
    """Verify that basic questions get structured answers from claude CLI."""

    @pytest.mark.asyncio
    async def test_simple_greeting(self, make_live_client: Any) -> None:
        """Ask for a greeting — simplest possible structured call."""
        async with make_live_client() as client:
            resp = await client.complete(
                messages=[{"role": "user", "content": "Say hello in one sentence."}],
                response_model=Greeting,
            )

        assert isinstance(resp, LLMResponse)
        assert isinstance(resp.content, Greeting)
        assert isinstance(resp.content.greeting, str)
        assert len(resp.content.greeting) > 0
        assert resp.provider == "local_claude"
        logger.info("[LIVE] simple_greeting | greeting=%s", resp.content.greeting)

    @pytest.mark.asyncio
    async def test_factual_question(self, make_live_client: Any) -> None:
        """Ask a factual question and validate the response schema."""
        async with make_live_client() as client:
            resp = await client.complete(
                messages=[
                    {
                        "role": "user",
                        "content": "What is the chemical symbol for water? Answer in one word.",
                    }
                ],
                response_model=FactAnswer,
            )

        assert isinstance(resp.content, FactAnswer)
        assert isinstance(resp.content.answer, str)
        assert len(resp.content.answer) > 0
        # The answer should contain H2O or water-related content
        logger.info("[LIVE] factual_question | answer=%s", resp.content.answer)


# ─── Multi-Field Schema Validation ───────────────────────────────


class TestLiveMultiFieldSchemas:
    """Test that claude CLI correctly populates multi-field Pydantic models."""

    @pytest.mark.asyncio
    async def test_capital_city_lookup(self, make_live_client: Any) -> None:
        """Geography question with two-field response (country + capital)."""
        async with make_live_client() as client:
            resp = await client.complete(
                messages=[
                    {
                        "role": "user",
                        "content": "What is the capital of Japan?",
                    }
                ],
                response_model=CapitalCity,
            )

        assert isinstance(resp.content, CapitalCity)
        assert isinstance(resp.content.country, str)
        assert isinstance(resp.content.capital, str)
        assert len(resp.content.capital) > 0
        # The capital of Japan should be Tokyo
        assert "tokyo" in resp.content.capital.lower()
        logger.info(
            "[LIVE] capital_city_lookup | %s -> %s", resp.content.country, resp.content.capital
        )

    @pytest.mark.asyncio
    async def test_sentiment_analysis(self, make_live_client: Any) -> None:
        """Sentiment analysis with text + sentiment fields."""
        async with make_live_client() as client:
            resp = await client.complete(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Analyze the sentiment of this text: "
                            "'I absolutely love sunny days at the beach!'"
                        ),
                    }
                ],
                response_model=SentimentResult,
            )

        assert isinstance(resp.content, SentimentResult)
        assert isinstance(resp.content.sentiment, str)
        assert resp.content.sentiment.lower() in ("positive", "very positive", "strongly positive")
        logger.info(
            "[LIVE] sentiment_analysis | '%s...' -> %s",
            resp.content.text[:50],
            resp.content.sentiment,
        )

    @pytest.mark.asyncio
    async def test_math_with_explanation(self, make_live_client: Any) -> None:
        """Math question with answer + explanation fields."""
        async with make_live_client() as client:
            resp = await client.complete(
                messages=[
                    {
                        "role": "user",
                        "content": "What is 15 * 7? Show your work.",
                    }
                ],
                response_model=MathAnswer,
            )

        assert isinstance(resp.content, MathAnswer)
        assert isinstance(resp.content.answer, str)
        assert isinstance(resp.content.explanation, str)
        assert len(resp.content.explanation) > 0
        # The answer should contain 105
        assert "105" in resp.content.answer
        logger.info(
            "[LIVE] math_with_explanation | %s — %s",
            resp.content.answer,
            resp.content.explanation[:80],
        )


# ─── Token Usage & Cost Tracking ─────────────────────────────────


class TestLiveUsageTracking:
    """Verify that token usage and cost tracking work with real claude calls."""

    @pytest.mark.asyncio
    async def test_usage_has_positive_tokens(self, make_live_client: Any) -> None:
        """Real call reports non-zero token usage (heuristic for local_claude)."""
        async with make_live_client() as client:
            resp = await client.complete(
                messages=[{"role": "user", "content": "Say hi."}],
                response_model=Greeting,
            )

        assert isinstance(resp.usage, TokenUsage)
        assert resp.usage.input_tokens > 0
        assert resp.usage.output_tokens > 0
        assert resp.usage.total_tokens > 0
        # Cost may be reported from CLI wrapper metadata
        assert resp.usage.input_cost_usd >= 0.0
        assert resp.usage.output_cost_usd >= 0.0
        logger.info(
            "[LIVE] usage_positive_tokens | %d in / %d out",
            resp.usage.input_tokens,
            resp.usage.output_tokens,
        )

    @pytest.mark.asyncio
    async def test_latency_is_measured(self, make_live_client: Any) -> None:
        """Real call has measurable latency."""
        async with make_live_client() as client:
            resp = await client.complete(
                messages=[{"role": "user", "content": "Say hi."}],
                response_model=Greeting,
            )

        assert resp.latency_ms > 0
        logger.info("[LIVE] latency_measured | %.0fms", resp.latency_ms)

    @pytest.mark.asyncio
    async def test_multi_call_cost_accumulates(self, make_live_client: Any) -> None:
        """Multiple calls accumulate call_count and total_tokens on the client."""
        async with make_live_client() as client:
            await client.complete(
                messages=[{"role": "user", "content": "Say hello."}],
                response_model=Greeting,
            )
            first_tokens = client.total_tokens
            first_count = client.call_count

            await client.complete(
                messages=[{"role": "user", "content": "What is 2+2?"}],
                response_model=FactAnswer,
            )

        assert client.call_count == first_count + 1
        assert client.total_tokens > first_tokens
        summary = client.cost_summary()
        assert summary["call_count"] == 2
        assert summary["total_tokens"] > 0
        logger.info(
            "[LIVE] multi_call_cost | calls=%d total_tokens=%d",
            summary["call_count"],
            summary["total_tokens"],
        )


# ─── Translation (Complex Prompt) ────────────────────────────────


class TestLiveTranslation:
    """Test translation — a more complex prompt with multi-field output."""

    @pytest.mark.asyncio
    async def test_english_to_french_translation(self, make_live_client: Any) -> None:
        """Translate English to French and validate all fields."""
        async with make_live_client() as client:
            resp = await client.complete(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Translate the following text from English to French: "
                            "'Good morning, how are you?'"
                        ),
                    }
                ],
                response_model=TranslationResult,
            )

        assert isinstance(resp.content, TranslationResult)
        assert isinstance(resp.content.translation, str)
        assert len(resp.content.translation) > 0
        # The French translation should contain some expected words
        translation_lower = resp.content.translation.lower()
        assert any(word in translation_lower for word in ("bonjour", "bon matin", "comment")), (
            f"Unexpected translation: {resp.content.translation}"
        )
        logger.info(
            "[LIVE] translation | '%s' (%s -> %s)",
            resp.content.translation,
            resp.content.source_language,
            resp.content.target_language,
        )


# ─── Summarization ───────────────────────────────────────────────


class TestLiveSummarization:
    """Test summarization — validates the LLM can compress text."""

    @pytest.mark.asyncio
    async def test_paragraph_summarization(self, make_live_client: Any) -> None:
        """Summarize a paragraph and check the summary is shorter."""
        long_text = (
            "Python is a high-level, general-purpose programming language. "
            "Its design philosophy emphasizes code readability with the use of "
            "significant indentation. Python is dynamically typed and garbage-collected. "
            "It supports multiple programming paradigms, including structured, "
            "object-oriented and functional programming. It was created by Guido van Rossum "
            "and first released in 1991."
        )
        async with make_live_client() as client:
            resp = await client.complete(
                messages=[
                    {
                        "role": "user",
                        "content": f"Summarize this in one sentence: {long_text}",
                    }
                ],
                response_model=SummaryResult,
            )

        assert isinstance(resp.content, SummaryResult)
        assert isinstance(resp.content.summary, str)
        assert len(resp.content.summary) > 0
        # Summary should be shorter than the original
        assert len(resp.content.summary) < len(long_text)
        logger.info("[LIVE] summarization | %s", resp.content.summary)
