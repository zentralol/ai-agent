"""Tests for the OpenAI-compatible chat model factory."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.llm import get_chat_model


@pytest.fixture(autouse=True)
def _clear_model_caches() -> None:
    get_settings.cache_clear()
    get_chat_model.cache_clear()


def test_get_chat_model_uses_non_streaming_provider_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "secret-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")

    model = get_chat_model()

    assert model is not None
    assert model.streaming is False
