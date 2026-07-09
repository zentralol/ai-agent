"""LLM client factory.

Builds an OpenAI-compatible chat model (DeepSeek/SenseNova/etc.) from settings.
Returns ``None`` when no API key is configured so callers can fall back to a
deterministic response instead of failing.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.config import get_settings

DEFAULT_TEMPERATURE = 0.7


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI | None:
    """Return a cached chat model, or ``None`` if unconfigured."""

    settings = get_settings()
    if not settings.llm_api_key:
        return None

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=SecretStr(settings.llm_api_key),
        base_url=settings.llm_base_url,
        # SenseNova/DeepSeek-compatible streamed tool calls can omit name/id chunks.
        # Keep model calls non-streaming; the API still streams typed SSE events.
        streaming=False,
        temperature=DEFAULT_TEMPERATURE,
    )
