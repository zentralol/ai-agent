"""Tests for the backend-driven place recommendations tool."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

import app.tools.recommendations_itinerary as recommendations_module
from app.config import Settings
from app.schemas.tools import ToolResponse, ToolStatus
from app.tools.recommendations_itinerary import (
    RecommendationsTool,
    get_place_recommendations,
)

_FULL_RECOMMEND_RESPONSE: dict[str, Any] = {
    "recommendations": [
        {
            "id": "fort-tryon",
            "name": "Fort Tryon Park",
            "lat": 40.8617,
            "lon": -73.9326,
            "neighborhood": "Washington Heights",
            "category": "park",
            "crowd_category": "Very quiet",
            "hours": "Open until 1:00 AM",
        }
    ],
    "based_on": "Your interests in quiet parks.",
}

_RECOMMEND_RESPONSE: dict[str, Any] = {
    "recommendations": [{"name": "MoMA"}],
    "based_on": "Your interests in art.",
}


def _settings_with_backend() -> Settings:
    return Settings(
        _env_file=None,
        BACKEND_API_BASE_URL="http://test-backend",
    )  # type: ignore[call-arg]


def _mock_tool(handler: Any) -> RecommendationsTool:
    tool = RecommendationsTool(_settings_with_backend())
    tool._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return tool


@pytest.mark.asyncio
async def test_recommend_rejects_non_ascii_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    class _FakeTool:
        async def recommend(self, **kwargs: Any) -> ToolResponse:
            nonlocal called
            called = True
            return ToolResponse(status=ToolStatus.SUCCESS, summary="", data={})

    monkeypatch.setattr(recommendations_module, "get_recommendations_tool", lambda: _FakeTool())

    raw = await get_place_recommendations.ainvoke(
        {"query": "安静的艺术馆"},
        config={"configurable": {"user_id": "u1"}},
    )
    result = ToolResponse.model_validate_json(raw)

    assert result.status == ToolStatus.WARNING
    assert "ASCII" in result.summary
    assert called is False


@pytest.mark.asyncio
async def test_recommend_allows_empty_query_with_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeTool:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def recommend(self, **kwargs: Any) -> ToolResponse:
            self.calls.append(kwargs)
            return ToolResponse(
                status=ToolStatus.SUCCESS,
                summary="1 recommendations returned. Art museums.",
                data=_RECOMMEND_RESPONSE,
            )

    fake = _FakeTool()
    monkeypatch.setattr(recommendations_module, "get_recommendations_tool", lambda: fake)

    raw = await get_place_recommendations.ainvoke(
        {"query": "", "category": "museum"},
        config={"configurable": {"user_id": "u1"}},
    )
    result = ToolResponse.model_validate_json(raw)

    assert result.status == ToolStatus.SUCCESS
    assert fake.calls == [
        {
            "user_id": "u1",
            "query": "",
            "category": "museum",
            "budget": None,
            "count": 6,
        }
    ]


@pytest.mark.asyncio
async def test_recommend_proceeds_with_ascii_query() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_RECOMMEND_RESPONSE)

    tool = _mock_tool(handler)
    result = await tool.recommend(
        user_id=None,
        query="quiet art museums",
        category=None,
        budget=None,
        count=6,
    )

    assert result.status == ToolStatus.SUCCESS
    assert "1 recommendations" in result.summary


@pytest.mark.asyncio
async def test_recommend_shapes_navigable_candidates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_FULL_RECOMMEND_RESPONSE)

    tool = _mock_tool(handler)
    result = await tool.recommend(
        user_id=None,
        query="quiet parks",
        category=None,
        budget=None,
        count=6,
    )

    assert result.status == ToolStatus.SUCCESS
    assert result.data["candidates"] == [
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
    ]
    assert result.data["recommendations"][0]["candidate_id"] == "recommend:fort-tryon"
