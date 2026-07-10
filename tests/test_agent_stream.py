"""Endpoint tests for /health and the streaming chat endpoint.

The chat model dependency is overridden with fakes so tests never hit a network.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import orjson
import pytest
from fastapi.testclient import TestClient
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field

import app.tools.preferences as preference_tools
from app.llm import get_chat_model
from app.main import app
from app.schemas.events import EventType
from app.schemas.tools import ToolResponse, ToolStatus
from app.tools.preferences import GET_USER_PREFERENCES_TOOL_NAME

client = TestClient(app)

_ToolDefinition = dict[str, Any] | type | Callable[..., Any] | BaseTool


class _FakeModel(FakeMessagesListChatModel):
    """Minimal stand-in for a LangChain chat model."""

    bound_tools: list[object] | None = None
    messages_by_call: list[list[BaseMessage]] = Field(default_factory=list)

    def __init__(
        self,
        responses: list[AIMessage] | None = None,
    ) -> None:
        model_responses: list[BaseMessage] = (
            list(responses) if responses is not None else [AIMessage(content="Hello there!")]
        )
        super().__init__(responses=model_responses)

    def bind_tools(
        self,
        tools: Sequence[_ToolDefinition],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> _FakeModel:
        self.bound_tools = list(tools)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.messages_by_call.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class _FailingModel(FakeMessagesListChatModel):
    def __init__(self) -> None:
        responses: list[BaseMessage] = [AIMessage(content="")]
        super().__init__(responses=responses)

    def bind_tools(
        self,
        tools: Sequence[_ToolDefinition],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> _FailingModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise RuntimeError("boom")


class _ChunkedToolCallFakeModel(BaseChatModel):
    """Streams one tool call across multiple chunks the way SenseNova/DeepSeek do:
    only the first chunk carries the tool name and id, continuation chunks carry
    only argument fragments. Regression coverage for the message corruption that
    LangGraph's experimental v3 astream_events protocol used to introduce when
    reconstructing such chunked tool calls (see runner.py's version note).
    """

    call_count: int = 0

    def bind_tools(
        self,
        tools: Sequence[_ToolDefinition],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> _ChunkedToolCallFakeModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("this fake model only supports streaming")

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        self.call_count += 1
        if self.call_count == 1:
            chunks = [
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": GET_USER_PREFERENCES_TOOL_NAME,
                            "args": "",
                            "id": "call-chunked",
                            "index": 0,
                            "type": "tool_call_chunk",
                        }
                    ],
                ),
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": "",
                            "args": "{",
                            "id": "",
                            "index": 0,
                            "type": "tool_call_chunk",
                        }
                    ],
                ),
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": "",
                            "args": "}",
                            "id": "",
                            "index": 0,
                            "type": "tool_call_chunk",
                        }
                    ],
                ),
            ]
        else:
            chunks = [AIMessageChunk(content="Personalized response")]
        for chunk in chunks:
            yield ChatGenerationChunk(message=chunk)

    @property
    def _llm_type(self) -> str:
        return "chunked-tool-call-fake"


class _FakePreferenceTool:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def get_user_preferences(self, user_id: str) -> ToolResponse:
        self.calls.append(user_id)
        return ToolResponse(
            status=ToolStatus.SUCCESS,
            summary="Loaded user preferences.",
            data={
                "preferences": {
                    "travel_pace": "relaxed",
                    "crowd_tolerance": "avoid",
                    "budget_range": "moderate",
                    "interests": ["parks"],
                    "mobility_needs": ["stepFree"],
                    "dietary_needs": [],
                    "inclusion_needs": ["quietSpaces"],
                    "onboarding_completed": True,
                },
                "source": "test",
            },
        )


def _override_model(model: object) -> Iterator[None]:
    app.dependency_overrides[get_chat_model] = lambda: model
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_chat_model, None)


@pytest.fixture
def fake_llm() -> Iterator[_FakeModel]:
    model = _FakeModel()
    with _dependency_override(get_chat_model, model):
        yield model


@pytest.fixture
def no_llm() -> Iterator[None]:
    yield from _override_model(None)


@pytest.fixture
def failing_llm() -> Iterator[None]:
    yield from _override_model(_FailingModel())


@pytest.fixture
def fake_preference_tool(monkeypatch: pytest.MonkeyPatch) -> _FakePreferenceTool:
    tool = _FakePreferenceTool()
    monkeypatch.setattr(preference_tools, "get_user_preference_tool", lambda: tool)
    return tool


@contextmanager
def _dependency_override(dependency: Callable[..., object], value: object) -> Iterator[None]:
    app.dependency_overrides[dependency] = lambda: value
    try:
        yield
    finally:
        app.dependency_overrides.pop(dependency, None)


def _parse_sse(body: str) -> list[dict[str, object]]:
    return [
        orjson.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def _assert_sequence(events: list[dict[str, object]]) -> None:
    assert [event.get("sequence") for event in events] == list(range(1, len(events) + 1))


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


def test_stream_llm_direct_answer_becomes_delta(fake_llm: _FakeModel) -> None:
    response = client.post("/api/v1/agent/stream", json=_valid_payload())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    _assert_sequence(events)
    deltas = [e for e in events if e["type"] == EventType.MESSAGE_DELTA.value]
    assert "".join(str(e["text"]) for e in deltas) == "Hello there!"
    assert events[-1]["type"] == EventType.DONE.value
    assert events[-1]["conversation_id"] == "conv-9"
    assert fake_llm.bound_tools is not None


def test_stream_loads_preferences_when_model_requests_tool(
    fake_preference_tool: _FakePreferenceTool,
) -> None:
    tool_call: dict[str, object] = {
        "name": GET_USER_PREFERENCES_TOOL_NAME,
        "args": {},
        "id": "call-pref",
    }
    model = _FakeModel(
        responses=[
            AIMessage(content="", tool_calls=[tool_call]),
            AIMessage(content="Personalized response"),
        ]
    )

    with _dependency_override(get_chat_model, model):
        response = client.post("/api/v1/agent/stream", json=_valid_payload())

    assert response.status_code == 200
    assert fake_preference_tool.calls
    assert fake_preference_tool.calls[0] == "u1"

    events = _parse_sse(response.text)
    _assert_sequence(events)
    assert events[0] == {
        "type": EventType.TOOL_STARTED.value,
        "tool_name": "get_user_preferences",
        "sequence": 1,
    }
    assert events[1]["type"] == EventType.TOOL_FINISHED.value
    assert events[1]["tool_name"] == "get_user_preferences"
    assert events[1]["tool_call_id"] == "call-pref"

    deltas = [e for e in events if e["type"] == EventType.MESSAGE_DELTA.value]
    assert "".join(str(e["text"]) for e in deltas) == "Personalized response"

    assert len(model.messages_by_call) == 2
    second_call_messages = model.messages_by_call[1]
    tool_messages = [
        message for message in second_call_messages if isinstance(message, ToolMessage)
    ]
    assert tool_messages
    assert '"crowd_tolerance":"avoid"' in str(tool_messages[0].content)


def test_stream_reconstructs_tool_call_streamed_across_multiple_chunks(
    fake_preference_tool: _FakePreferenceTool,
) -> None:
    model = _ChunkedToolCallFakeModel()

    with _dependency_override(get_chat_model, model):
        response = client.post("/api/v1/agent/stream", json=_valid_payload())

    assert response.status_code == 200
    events = _parse_sse(response.text)
    _assert_sequence(events)

    assert events[0]["type"] == EventType.TOOL_STARTED.value
    assert events[0]["tool_name"] == "get_user_preferences"

    assert events[1]["type"] == EventType.TOOL_FINISHED.value
    assert events[1]["tool_name"] == "get_user_preferences"
    assert events[1]["tool_call_id"] == "call-chunked"

    deltas = [e for e in events if e["type"] == EventType.MESSAGE_DELTA.value]
    assert "".join(str(e["text"]) for e in deltas) == "Personalized response"

    assert fake_preference_tool.calls
    assert fake_preference_tool.calls[0] == "u1"


def test_stream_does_not_load_preferences_without_model_tool_call(
    fake_llm: _FakeModel, fake_preference_tool: _FakePreferenceTool
) -> None:
    response = client.post("/api/v1/agent/stream", json=_valid_payload())

    assert response.status_code == 200
    assert fake_preference_tool.calls == []


def test_stream_fallback_when_no_llm(no_llm: None) -> None:
    response = client.post("/api/v1/agent/stream", json=_valid_payload())

    assert response.status_code == 200
    events = _parse_sse(response.text)
    _assert_sequence(events)
    assert events[0]["type"] == EventType.WARNING.value
    assert any(e["type"] == EventType.MESSAGE_DELTA.value for e in events)
    assert events[-1]["type"] == EventType.DONE.value


def test_stream_error_event_on_llm_failure(failing_llm: None) -> None:
    response = client.post("/api/v1/agent/stream", json=_valid_payload())

    assert response.status_code == 200
    events = _parse_sse(response.text)
    _assert_sequence(events)
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
