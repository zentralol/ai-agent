"""Tests for the Google Places nearby-places tool."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

import app.tools.places as places_module
from app.config import Settings
from app.schemas.tools import ToolResponse, ToolStatus
from app.tools.places import (
    PLACES_FIELD_MASK,
    PLACES_SEARCH_TEXT_URL,
    PlacesTool,
    get_nearby_places,
)

ORIGIN = {"lat": 40.7580, "lng": -73.9855}

_PLACES_RESPONSE: dict[str, Any] = {
    "places": [
        {
            "displayName": {"text": "Blue Bottle Coffee"},
            "formattedAddress": "1 Rockefeller Plaza, New York",
            "location": {"latitude": 40.7585, "longitude": -73.9860},
            "primaryTypeDisplayName": {"text": "Coffee shop"},
            "rating": 4.5,
            "userRatingCount": 200,
            "currentOpeningHours": {"openNow": True},
            "priceLevel": "PRICE_LEVEL_MODERATE",
        }
    ]
}


def _settings_with_key() -> Settings:
    return Settings(_env_file=None, GOOGLE_MAPS_API_KEY="test-key")  # type: ignore[call-arg]


def _mock_tool(handler: Any) -> PlacesTool:
    tool = PlacesTool(_settings_with_key())
    tool._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return tool


def test_tool_exposes_only_query_argument() -> None:
    # query is model-supplied; config (and thus lat/lng) is injected, not an arg.
    assert set(get_nearby_places.args) == {"query"}


@pytest.mark.asyncio
async def test_search_builds_request_and_shapes_results() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json=_PLACES_RESPONSE)

    tool = _mock_tool(handler)
    result = await tool.search_nearby("coffee", ORIGIN["lat"], ORIGIN["lng"])

    # Request shape.
    request = captured["request"]
    assert str(request.url) == PLACES_SEARCH_TEXT_URL
    assert request.headers["X-Goog-Api-Key"] == "test-key"
    assert request.headers["X-Goog-FieldMask"] == PLACES_FIELD_MASK
    body = json.loads(request.content)
    assert body["textQuery"] == "coffee"
    assert body["locationBias"]["circle"]["center"] == {
        "latitude": ORIGIN["lat"],
        "longitude": ORIGIN["lng"],
    }

    # Result shape.
    assert result.status == ToolStatus.SUCCESS
    place = result.data["places"][0]
    assert place["name"] == "Blue Bottle Coffee"
    assert place["primary_type"] == "Coffee shop"
    assert place["rating"] == 4.5
    assert place["open_now"] is True
    assert place["price_level"] == "PRICE_LEVEL_MODERATE"
    assert isinstance(place["distance_km"], float)
    # The place's own (public) coordinates are returned for navigation.
    assert place["lat"] == 40.7585
    assert place["lng"] == -73.986

    # The user's coordinates are never returned: only places + query.
    assert set(result.data.keys()) == {"places", "query"}


@pytest.mark.asyncio
async def test_search_returns_error_on_http_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    tool = _mock_tool(handler)
    result = await tool.search_nearby("coffee", ORIGIN["lat"], ORIGIN["lng"])

    assert result.status == ToolStatus.ERROR


@pytest.mark.asyncio
async def test_search_warns_without_api_key() -> None:
    tool = PlacesTool(Settings(_env_file=None))  # type: ignore[call-arg]

    result = await tool.search_nearby("coffee", ORIGIN["lat"], ORIGIN["lng"])

    assert result.status == ToolStatus.WARNING


@pytest.mark.asyncio
async def test_tool_passes_query_and_device_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeTool:
        def __init__(self) -> None:
            self.calls: list[tuple[str, float, float]] = []

        async def search_nearby(self, query: str, lat: float, lng: float) -> ToolResponse:
            self.calls.append((query, lat, lng))
            return ToolResponse(
                status=ToolStatus.SUCCESS,
                summary="Found 1 places for 'coffee'.",
                data={"places": [{"name": "X"}], "query": query},
            )

    fake = _FakeTool()
    monkeypatch.setattr(places_module, "get_places_tool", lambda: fake)

    raw = await get_nearby_places.ainvoke(
        {"query": "coffee"},
        config={"configurable": {"lat": ORIGIN["lat"], "lng": ORIGIN["lng"]}},
    )
    result = ToolResponse.model_validate_json(raw)

    assert result.status == ToolStatus.SUCCESS
    assert fake.calls == [("coffee", ORIGIN["lat"], ORIGIN["lng"])]


@pytest.mark.asyncio
async def test_tool_warns_without_query(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    class _FakeTool:
        async def search_nearby(self, query: str, lat: float, lng: float) -> ToolResponse:
            nonlocal called
            called = True
            return ToolResponse(status=ToolStatus.SUCCESS, summary="", data={})

    monkeypatch.setattr(places_module, "get_places_tool", lambda: _FakeTool())

    raw = await get_nearby_places.ainvoke(
        {"query": "   "},
        config={"configurable": {"lat": ORIGIN["lat"], "lng": ORIGIN["lng"]}},
    )
    result = ToolResponse.model_validate_json(raw)

    assert result.status == ToolStatus.WARNING
    assert called is False


@pytest.mark.asyncio
async def test_tool_warns_without_location(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    class _FakeTool:
        async def search_nearby(self, query: str, lat: float, lng: float) -> ToolResponse:
            nonlocal called
            called = True
            return ToolResponse(status=ToolStatus.SUCCESS, summary="", data={})

    monkeypatch.setattr(places_module, "get_places_tool", lambda: _FakeTool())

    raw = await get_nearby_places.ainvoke(
        {"query": "coffee"},
        config={"configurable": {"user_id": "u1"}},
    )
    result = ToolResponse.model_validate_json(raw)

    assert result.status == ToolStatus.WARNING
    assert called is False
