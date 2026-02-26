"""Pydantic response models used across integration tests.

These represent the kind of structured outputs a real consumer project
(e.g., job-hunter-agent) would define and pass to LLMClient.complete().
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FactAnswer(BaseModel):
    """Simple factual Q&A response."""

    answer: str = Field(description="The factual answer to the question.")


class CapitalCity(BaseModel):
    """Geography knowledge response."""

    country: str = Field(description="The country asked about.")
    capital: str = Field(description="The capital city of that country.")


class SentimentResult(BaseModel):
    """Sentiment analysis response."""

    text: str = Field(description="The original text that was analyzed.")
    sentiment: str = Field(description="The detected sentiment: positive, negative, or neutral.")


class TranslationResult(BaseModel):
    """Translation response."""

    source_language: str = Field(description="The detected or specified source language.")
    target_language: str = Field(description="The target language for translation.")
    translation: str = Field(description="The translated text.")


class SummaryResult(BaseModel):
    """Text summarization response."""

    summary: str = Field(description="A concise summary of the input text.")


class MathAnswer(BaseModel):
    """Math problem response."""

    answer: str = Field(description="The numerical or symbolic answer.")
    explanation: str = Field(description="Step-by-step explanation of the solution.")


class Greeting(BaseModel):
    """Simple greeting response."""

    greeting: str = Field(description="A friendly greeting message.")
