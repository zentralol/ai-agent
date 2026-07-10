"""Tests for inbound internal service-to-service authentication.

The agent router is protected by ``require_internal_auth``. These tests drive the
public endpoint through :class:`TestClient`, overriding the settings dependency to
control whether a token is configured, and the chat model dependency so no network
call is made once auth passes.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.config import get_settings
from app.llm import get_chat_model
from app.main import app

client = TestClient(app)

CONFIGURED_TOKEN = "s3cret-internal-token"


@contextmanager
def _override(dependency: object, value: object) -> Iterator[None]:
    app.dependency_overrides[dependency] = lambda: value
    try:
        yield
    finally:
        app.dependency_overrides.pop(dependency, None)


@contextmanager
def _token_configured(token: str | None) -> Iterator[None]:
    settings = SimpleNamespace(agent_internal_token=token)
    with _override(get_settings, settings):
        yield


def _valid_payload() -> dict[str, object]:
    return {
        "user_id": "u1",
        "message": "hi",
        "client_type": "web",
        "conversation_id": "conv-1",
    }


def test_rejects_missing_token_when_configured() -> None:
    with _token_configured(CONFIGURED_TOKEN):
        response = client.post("/api/v1/agent/stream", json=_valid_payload())

    assert response.status_code == 401


def test_rejects_wrong_token_when_configured() -> None:
    with _token_configured(CONFIGURED_TOKEN):
        response = client.post(
            "/api/v1/agent/stream",
            json=_valid_payload(),
            headers={"X-Internal-Service-Token": "wrong-token"},
        )

    assert response.status_code == 401


def test_accepts_correct_token_when_configured() -> None:
    # LLM disabled -> deterministic fallback stream once auth passes.
    with _token_configured(CONFIGURED_TOKEN), _override(get_chat_model, None):
        response = client.post(
            "/api/v1/agent/stream",
            json=_valid_payload(),
            headers={"X-Internal-Service-Token": CONFIGURED_TOKEN},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_allows_request_when_token_unconfigured() -> None:
    with _token_configured(None), _override(get_chat_model, None):
        response = client.post("/api/v1/agent/stream", json=_valid_payload())

    assert response.status_code == 200


def test_health_is_not_protected() -> None:
    with _token_configured(CONFIGURED_TOKEN):
        response = client.get("/health")

    assert response.status_code == 200
