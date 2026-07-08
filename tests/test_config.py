"""Tests for the environment-driven settings loader."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()


def test_defaults_applied_when_env_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: ensure no relevant env vars leak in from the host.
    for key in ("APP_ENV", "PORT", "LLM_PROVIDER", "LLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    # Act
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    # Assert
    assert settings.app_env == "development"
    assert settings.port == 8010
    assert settings.llm_provider == "deepseek"
    assert settings.llm_api_key is None


def test_reads_and_coerces_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("LLM_API_KEY", "secret-key")

    # Act
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    # Assert
    assert settings.app_env == "production"
    assert settings.port == 9000
    assert settings.llm_api_key == "secret-key"


def test_settings_are_frozen() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        settings.port = 1234  # type: ignore[misc]


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
