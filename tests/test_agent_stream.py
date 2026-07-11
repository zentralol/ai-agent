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

import app.tools.itinerary as itinerary_module
import app.tools.places as places_module
import app.tools.preferences as preference_tools
import app.tools.recommendations_itinerary as recommendations_module
from app.llm import get_chat_model
from app.main import app
from app.schemas.events import EventType
from app.schemas.tools import ToolResponse, ToolStatus
from app.tools.itinerary import PLAN_ITINERARY_TOOL_NAME
from app.tools.places import GET_NEARBY_PLACES_TOOL_NAME
from app.tools.preferences import GET_USER_PREFERENCES_TOOL_NAME
from app.tools.recommendations import SELECT_RECOMMENDED_PLACES_TOOL_NAME
from app.tools.recommendations_itinerary import RECOMMEND_TOOL_NAME

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


class _FakePlacesTool:
    async def search_nearby(self, query: str, lat: float, lng: float) -> ToolResponse:
        return ToolResponse(
            status=ToolStatus.SUCCESS,
            summary=f"Found places for {query}.",
            data={
                "places": [
                    {
                        "candidate_id": "google:place-a",
                        "name": "Place A",
                        "address": "1 Main St",
                        "primary_type": "Cafe",
                        "lat": 40.7,
                        "lng": -73.9,
                        "rating": 4.5,
                        "distance_km": 0.2,
                    },
                    {
                        "candidate_id": "google:place-b",
                        "name": "Place B",
                        "address": "2 Main St",
                        "primary_type": "Cafe",
                        "lat": 40.71,
                        "lng": -73.91,
                        "rating": 4.0,
                        "distance_km": 0.4,
                    },
                ],
                "query": query,
            },
        )


class _FakeRecommendationsTool:
    async def recommend(self, **kwargs: Any) -> ToolResponse:
        return ToolResponse(
            status=ToolStatus.SUCCESS,
            summary="2 recommendations returned.",
            data={
                "recommendations": [
                    {
                        "id": "fort-tryon",
                        "candidate_id": "recommend:fort-tryon",
                        "name": "Fort Tryon Park",
                        "lat": 40.8617,
                        "lon": -73.9326,
                        "neighborhood": "Washington Heights",
                        "category": "park",
                        "crowd_category": "Very quiet",
                        "hours": "Open until 1:00 AM",
                    }
                ],
                "candidates": [
                    {
                        "candidate_id": "recommend:fort-tryon",
                        "name": "Fort Tryon Park",
                        "lat": 40.8617,
                        "lng": -73.9326,
                        "neighborhood": "Washington Heights",
                        "category": "park",
                        "crowd_category": "Very quiet",
                        "hours": "Open until 1:00 AM",
                    }
                ],
                "based_on": "Quiet parks.",
            },
        )


class _FakeItineraryTool:
    async def plan(self, **kwargs: Any) -> ToolResponse:
        return ToolResponse(
            status=ToolStatus.SUCCESS,
            summary="Itinerary built: 2 stops starting at 16:00.",
            data={
                "stops": [
                    {
                        "time": "16:00",
                        "place_id": "washington-square",
                        "place_name": "Washington Square Park",
                        "candidate_id": "itinerary:washington-square",
                        "lat": 40.7308,
                        "lon": -73.9973,
                        "neighborhood": "Greenwich Village",
                        "category": "park",
                        "crowd_category": "Very busy",
                        "hours": "Open 24 hours",
                        "why_recommended": "Historic park stroll",
                    },
                    {
                        "time": "20:10",
                        "place_id": "essex-market",
                        "place_name": "Essex Market",
                        "candidate_id": "itinerary:essex-market",
                        "lat": 40.7185,
                        "lon": -73.9877,
                        "neighborhood": "Lower East Side",
                        "category": "food",
                        "crowd_category": "Moderate",
                        "hours": "08:00-21:00",
                        "why_recommended": "Vegetarian-friendly dinner",
                    },
                ],
                "candidates": [
                    {
                        "candidate_id": "itinerary:washington-square",
                        "name": "Washington Square Park",
                        "lat": 40.7308,
                        "lng": -73.9973,
                        "time": "16:00",
                        "neighborhood": "Greenwich Village",
                        "category": "park",
                        "crowd_category": "Very busy",
                        "hours": "Open 24 hours",
                        "why_recommended": "Historic park stroll",
                    },
                    {
                        "candidate_id": "itinerary:essex-market",
                        "name": "Essex Market",
                        "lat": 40.7185,
                        "lng": -73.9877,
                        "time": "20:10",
                        "neighborhood": "Lower East Side",
                        "category": "food",
                        "crowd_category": "Moderate",
                        "hours": "08:00-21:00",
                        "why_recommended": "Vegetarian-friendly dinner",
                    },
                ],
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


def test_stream_emits_only_structured_recommendations_in_selection_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(places_module, "get_places_tool", lambda: _FakePlacesTool())
    model = _FakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": GET_NEARBY_PLACES_TOOL_NAME,
                        "args": {"query": "coffee"},
                        "id": "call-search",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": SELECT_RECOMMENDED_PLACES_TOOL_NAME,
                        "args": {
                            "recommendations": [
                                {
                                    "candidate_id": "google:place-b",
                                    "reason": "Quieter",
                                }
                            ]
                        },
                        "id": "call-select",
                    }
                ],
            ),
            AIMessage(content="I recommend Place B."),
        ]
    )

    with _dependency_override(get_chat_model, model):
        response = client.post(
            "/api/v1/agent/stream",
            json={**_valid_payload(), "lat": 40.7, "lng": -73.9},
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    _assert_sequence(events)
    recommendation_events = [
        event for event in events if event["type"] == "recommendations"
    ]
    assert len(recommendation_events) == 1
    assert recommendation_events[0]["data"]["items"] == [
        {
            "candidate_id": "google:place-b",
            "source": "nearby",
            "name": "Place B",
            "lat": 40.71,
            "lng": -73.91,
            "subtitle": "2 Main St",
            "detail": "Cafe · ★ 4.0 · 0.4 km",
            "rank": 1,
            "reason": "Quieter",
        }
    ]


def test_stream_emits_recommendations_from_backend_place_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recommendations_module,
        "get_recommendations_tool",
        lambda: _FakeRecommendationsTool(),
    )
    model = _FakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": RECOMMEND_TOOL_NAME,
                        "args": {"query": "quiet parks tonight"},
                        "id": "call-recommend",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": SELECT_RECOMMENDED_PLACES_TOOL_NAME,
                        "args": {
                            "recommendations": [
                                {
                                    "candidate_id": "recommend:fort-tryon",
                                    "reason": "Very quiet",
                                }
                            ]
                        },
                        "id": "call-select",
                    }
                ],
            ),
            AIMessage(content="I recommend Fort Tryon Park."),
        ]
    )

    with _dependency_override(get_chat_model, model):
        response = client.post("/api/v1/agent/stream", json=_valid_payload())

    assert response.status_code == 200
    events = _parse_sse(response.text)
    recommendation_events = [
        event for event in events if event["type"] == "recommendations"
    ]
    assert len(recommendation_events) == 1
    assert recommendation_events[0]["data"]["items"] == [
        {
            "candidate_id": "recommend:fort-tryon",
            "source": "recommend",
            "name": "Fort Tryon Park",
            "lat": 40.8617,
            "lng": -73.9326,
            "subtitle": "Washington Heights",
            "detail": "park · Very quiet · Open until 1:00 AM",
            "rank": 1,
            "reason": "Very quiet",
        }
    ]


def test_stream_backfills_recommendations_when_selection_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recommendations_module,
        "get_recommendations_tool",
        lambda: _FakeRecommendationsTool(),
    )
    model = _FakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": RECOMMEND_TOOL_NAME,
                        "args": {"query": "quiet parks tonight"},
                        "id": "call-recommend",
                    }
                ],
            ),
            AIMessage(content="Fort Tryon Park is a peaceful escape tonight."),
        ]
    )

    with _dependency_override(get_chat_model, model):
        response = client.post("/api/v1/agent/stream", json=_valid_payload())

    assert response.status_code == 200
    events = _parse_sse(response.text)
    recommendation_events = [
        event for event in events if event["type"] == "recommendations"
    ]
    assert len(recommendation_events) == 1
    assert recommendation_events[0]["data"]["source"] == "recommend"
    assert recommendation_events[0]["data"]["items"][0]["name"] == "Fort Tryon Park"


def test_stream_emits_recommendations_from_itinerary_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        itinerary_module,
        "get_itinerary_tool",
        lambda: _FakeItineraryTool(),
    )
    model = _FakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": PLAN_ITINERARY_TOOL_NAME,
                        "args": {
                            "anchor_place": "Greenwich Village",
                            "anchor_time": "2026-07-10T16:00:00",
                            "duration_hours": 6,
                        },
                        "id": "call-itinerary",
                    }
                ],
            ),
            AIMessage(content="Here is your relaxed Greenwich Village evening."),
        ]
    )

    with _dependency_override(get_chat_model, model):
        response = client.post("/api/v1/agent/stream", json=_valid_payload())

    assert response.status_code == 200
    events = _parse_sse(response.text)
    recommendation_events = [
        event for event in events if event["type"] == "recommendations"
    ]
    assert len(recommendation_events) == 1
    assert recommendation_events[0]["data"]["source"] == "itinerary"
    assert [item["name"] for item in recommendation_events[0]["data"]["items"]] == [
        "Washington Square Park",
        "Essex Market",
    ]
    assert recommendation_events[0]["data"]["items"][0]["reason"] == (
        "Historic park stroll"
    )


def test_stream_attaches_itinerary_date_range_from_anchor_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        itinerary_module,
        "get_itinerary_tool",
        lambda: _FakeItineraryTool(),
    )
    model = _FakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": PLAN_ITINERARY_TOOL_NAME,
                        "args": {
                            "anchor_place": "Greenwich Village",
                            "anchor_time": "2026-07-10T16:00:00",
                            "duration_hours": 6,
                        },
                        "id": "call-itinerary",
                    }
                ],
            ),
            AIMessage(content="Here is your relaxed Greenwich Village evening."),
        ]
    )

    with _dependency_override(get_chat_model, model):
        response = client.post("/api/v1/agent/stream", json=_valid_payload())

    assert response.status_code == 200
    events = _parse_sse(response.text)
    recommendation_events = [
        event for event in events if event["type"] == "recommendations"
    ]
    assert len(recommendation_events) == 1
    data = recommendation_events[0]["data"]
    assert data["source"] == "itinerary"
    assert data["start_date"] == "2026-07-10"
    assert data["end_date"] == "2026-07-10"
    assert "Jul 10" in str(data["items"][0]["subtitle"])


def test_stream_attaches_plan_summary_from_tool_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        itinerary_module,
        "get_itinerary_tool",
        lambda: _FakeItineraryTool(),
    )
    plan_summary = (
        "A relaxed Greenwich Village evening: stroll through Washington Square "
        "Park, then grab bites at Essex Market."
    )
    model = _FakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": PLAN_ITINERARY_TOOL_NAME,
                        "args": {
                            "anchor_place": "Greenwich Village",
                            "anchor_time": "2026-07-10T16:00:00",
                            "duration_hours": 6,
                        },
                        "id": "call-itinerary",
                    }
                ],
            ),
            # The streamed chat reply the user sees (with its preamble).
            AIMessage(content="Sure! Here is your relaxed Greenwich Village evening."),
            # The follow-up, tool-free summarization call reads the plan output.
            AIMessage(content=plan_summary),
        ]
    )

    with _dependency_override(get_chat_model, model):
        response = client.post("/api/v1/agent/stream", json=_valid_payload())

    assert response.status_code == 200
    events = _parse_sse(response.text)
    recommendation_events = [
        event for event in events if event["type"] == "recommendations"
    ]
    assert len(recommendation_events) == 1
    # The saved summary is the dedicated plan summary, not the streamed preamble.
    assert recommendation_events[0]["data"]["summary"] == plan_summary


def test_stream_allows_many_tool_steps_for_multi_day_plans(
    fake_preference_tool: _FakePreferenceTool,
) -> None:
    # A multi-day plan calls the day-planning tool once per day, exceeding the
    # old 5-step ceiling. The turn must complete rather than hit the step limit.
    calls = 6
    model = _FakeModel(
        responses=[
            *(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": GET_USER_PREFERENCES_TOOL_NAME,
                            "args": {},
                            "id": f"call-{index}",
                        }
                    ],
                )
                for index in range(calls)
            ),
            AIMessage(content="Here is your multi-day plan."),
        ]
    )

    with _dependency_override(get_chat_model, model):
        response = client.post("/api/v1/agent/stream", json=_valid_payload())

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert not any(event["type"] == EventType.ERROR.value for event in events)
    deltas = [e for e in events if e["type"] == EventType.MESSAGE_DELTA.value]
    assert "".join(str(e["text"]) for e in deltas) == "Here is your multi-day plan."


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
