"""Endpoint tests for /health and the streaming chat endpoint.

The chat model dependency is overridden with fakes so tests never hit a network.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import orjson
import pytest
from fastapi.testclient import TestClient

from app.llm import get_chat_model
from app.main import app
from app.schemas.events import EventType

client = TestClient(app)


class _FakeChunk:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeModel:
    """Minimal stand-in for a LangChain chat model."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def astream(self, messages: object) -> AsyncIterator[_FakeChunk]:
        for text in self._chunks:
            yield _FakeChunk(text)


class _FailingModel:
    async def astream(self, messages: object) -> AsyncIterator[_FakeChunk]:
        raise RuntimeError("boom")
        yield _FakeChunk("")  # pragma: no cover - unreachable, makes this an async gen


def _override_model(model: object) -> Iterator[None]:
    app.dependency_overrides[get_chat_model] = lambda: model
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_chat_model, None)


@pytest.fixture
def fake_llm() -> Iterator[None]:
    yield from _override_model(_FakeModel(["Hello", " there", "!"]))


@pytest.fixture
def no_llm() -> Iterator[None]:
    yield from _override_model(None)


@pytest.fixture
def failing_llm() -> Iterator[None]:
    yield from _override_model(_FailingModel())


def _parse_sse(body: str) -> list[dict[str, object]]:
    return [
        orjson.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def _valid_payload() -> dict[str, object]:
    return {
        "user_id": "u1",
        "message": "hi",
        "client_type": "web",
        "conversation_id": "conv-9",
    }


def test_health_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "zentra-agent"}


def test_stream_llm_tokens_become_deltas(fake_llm: None) -> None:
    response = client.post("/api/v1/agent/stream", json=_valid_payload())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    deltas = [e for e in events if e["type"] == EventType.MESSAGE_DELTA.value]
    assert "".join(str(e["text"]) for e in deltas) == "Hello there!"
    assert events[-1]["type"] == EventType.DONE.value
    assert events[-1]["conversation_id"] == "conv-9"


def test_stream_fallback_when_no_llm(no_llm: None) -> None:
    response = client.post("/api/v1/agent/stream", json=_valid_payload())

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert events[0]["type"] == EventType.WARNING.value
    assert any(e["type"] == EventType.MESSAGE_DELTA.value for e in events)
    assert events[-1]["type"] == EventType.DONE.value


def test_stream_error_event_on_llm_failure(failing_llm: None) -> None:
    response = client.post("/api/v1/agent/stream", json=_valid_payload())

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert events[-1]["type"] == EventType.ERROR.value
    assert events[-1]["code"] == "LLM_ERROR"


def test_stream_invalid_request_returns_422() -> None:
    response = client.post(
        "/api/v1/agent/stream",
        json={"message": "hi", "client_type": "web"},
    )

    assert response.status_code == 422


def test_openapi_documents_stream_endpoint() -> None:
    schema = client.get("/openapi.json").json()

    assert "/api/v1/agent/stream" in schema["paths"]
