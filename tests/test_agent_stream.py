"""Endpoint tests for /health and the streaming chat endpoint."""

from __future__ import annotations

import orjson
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.events import EventType

client = TestClient(app)


def _parse_sse(body: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(orjson.loads(line[len("data: ") :]))
    return events


def test_health_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "zentra-agent"}


def test_stream_valid_request_returns_events() -> None:
    # Arrange
    payload = {
        "user_id": "u1",
        "message": "hi",
        "client_type": "web",
        "conversation_id": "conv-9",
    }

    # Act
    response = client.post("/api/v1/agent/stream", json=payload)

    # Assert
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    assert events, "expected at least one SSE event"
    assert all(e["type"] == EventType.MESSAGE_DELTA.value for e in events[:-1])
    assert events[-1]["type"] == EventType.DONE.value
    assert events[-1]["conversation_id"] == "conv-9"


def test_stream_invalid_request_returns_422() -> None:
    response = client.post(
        "/api/v1/agent/stream",
        json={"message": "hi", "client_type": "web"},
    )

    assert response.status_code == 422


def test_openapi_documents_stream_endpoint() -> None:
    schema = client.get("/openapi.json").json()

    assert "/api/v1/agent/stream" in schema["paths"]
