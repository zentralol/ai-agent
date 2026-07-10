"""Tests for the itinerary planning tool."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

import app.tools.itinerary as itinerary_module
from app.config import Settings
from app.schemas.tools import ToolResponse, ToolStatus
from app.tools.itinerary import ItineraryTool, plan_itinerary

_ITINERARY_RESPONSE: dict[str, Any] = {
    "stops": [
        {"time": "10:00", "place": "Central Park", "activity": "Walk"},
    ]
}

_FULL_ITINERARY_RESPONSE: dict[str, Any] = {
    "stops": [
        {
            "time": "16:00",
            "place_id": "washington-square",
            "place_name": "Washington Square Park",
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
            "lat": 40.7185,
            "lon": -73.9877,
            "neighborhood": "Lower East Side",
            "category": "food",
            "crowd_category": "Moderate",
            "hours": "08:00-21:00",
            "why_recommended": "Vegetarian-friendly dinner",
        },
    ]
}


def _settings_with_backend() -> Settings:
    return Settings(
        _env_file=None,
        BACKEND_API_BASE_URL="http://test-backend",
    )  # type: ignore[call-arg]


def _mock_tool(handler: Any) -> ItineraryTool:
    tool = ItineraryTool(_settings_with_backend())
    tool._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return tool


@pytest.mark.asyncio
async def test_plan_rejects_non_ascii_anchor_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    class _FakeTool:
        async def plan(self, **kwargs: Any) -> ToolResponse:
            nonlocal called
            called = True
            return ToolResponse(status=ToolStatus.SUCCESS, summary="", data={})

    monkeypatch.setattr(itinerary_module, "get_itinerary_tool", lambda: _FakeTool())

    raw = await plan_itinerary.ainvoke(
        {
            "anchor_place": "中央公园",
            "anchor_time": "2026-07-10T10:00:00",
        },
        config={"configurable": {"user_id": "u1"}},
    )
    result = ToolResponse.model_validate_json(raw)

    assert result.status == ToolStatus.WARNING
    assert "ASCII" in result.summary
    assert called is False


@pytest.mark.asyncio
async def test_plan_proceeds_with_ascii_anchor_place() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ITINERARY_RESPONSE)

    tool = _mock_tool(handler)
    result = await tool.plan(
        user_id=None,
        anchor_place="Central Park",
        anchor_time="2026-07-10T10:00:00",
        duration_hours=8,
        additional_context="",
    )

    assert result.status == ToolStatus.SUCCESS
    assert "1 stops" in result.summary


@pytest.mark.asyncio
async def test_tool_invocation_proceeds_with_ascii_anchor_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeTool:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def plan(self, **kwargs: Any) -> ToolResponse:
            self.calls.append(kwargs["anchor_place"])
            return ToolResponse(
                status=ToolStatus.SUCCESS,
                summary="Itinerary built: 1 stops starting at 10:00.",
                data=_ITINERARY_RESPONSE,
            )

    fake = _FakeTool()
    monkeypatch.setattr(itinerary_module, "get_itinerary_tool", lambda: fake)

    raw = await plan_itinerary.ainvoke(
        {
            "anchor_place": "Central Park",
            "anchor_time": "2026-07-10T10:00:00",
        },
        config={"configurable": {"user_id": "u1"}},
    )
    result = ToolResponse.model_validate_json(raw)

    assert result.status == ToolStatus.SUCCESS
    assert fake.calls == ["Central Park"]


@pytest.mark.asyncio
async def test_plan_shapes_navigable_candidates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_FULL_ITINERARY_RESPONSE)

    tool = _mock_tool(handler)
    result = await tool.plan(
        user_id=None,
        anchor_place="Greenwich Village",
        anchor_time="2026-07-10T16:00:00",
        duration_hours=6,
        additional_context="",
    )

    assert result.status == ToolStatus.SUCCESS
    assert result.data["candidates"] == [
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
    ]
    assert (
        result.data["stops"][0]["candidate_id"] == "itinerary:washington-square"
    )
