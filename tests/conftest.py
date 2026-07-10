"""Shared test fixtures.

Tests must not depend on a developer's local ``.env``. Real ``SUPABASE_*`` or
``AGENT_INTERNAL_TOKEN`` values there would otherwise activate inbound auth and
conversation persistence in tests that assume a clean environment. This fixture
forces those secrets empty (env vars take precedence over ``.env``, and the code
treats empty as unconfigured) and clears the cached settings/singletons so each
test starts from a hermetic baseline. Tests that exercise auth or persistence
override the relevant dependency explicitly.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app import config
from app.conversations import repository
from app.tools import preferences

_NEUTRALIZED_SECRETS = (
    "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
    "AGENT_INTERNAL_TOKEN",
)

_CACHED_SINGLETONS = (
    config.get_settings,
    repository.get_conversation_repository,
    preferences.get_user_preference_tool,
)


def _clear_caches() -> None:
    for cached in _CACHED_SINGLETONS:
        cached.cache_clear()


@pytest.fixture(autouse=True)
def hermetic_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in _NEUTRALIZED_SECRETS:
        monkeypatch.setenv(name, "")
    _clear_caches()
    yield
    _clear_caches()
