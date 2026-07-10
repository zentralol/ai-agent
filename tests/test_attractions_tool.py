"""Tests for the nearest-attractions tool."""

from __future__ import annotations

from typing import Any

import pytest

import app.tools.attractions as attractions_module
from app.config import Settings
from app.geo import haversine_km
from app.schemas.tools import ToolResponse, ToolStatus
from app.tools.attractions import (
    AttractionsTool,
    _rank_by_distance,
    get_nearest_attractions,
)

# Origin near Times Square.
ORIGIN = {"lat": 40.7580, "lng": -73.9855}


def _row(name: str, lat: float, lon: float, description: str = "A place.") -> dict[str, Any]:
    return {
        "id": 1,
        "Name": name,
        "Category": "Landmark",
        "Neighborhood": "Midtown",
        "Description": description,
        "lat": lat,
        "lon": lon,
    }


def test_tool_exposes_no_llm_arguments() -> None:
    # Location comes from runtime config, never from the model.
    assert get_nearest_attractions.args == {}


def test_haversine_zero_and_one_degree() -> None:
    assert haversine_km(40.0, -73.0, 40.0, -73.0) == pytest.approx(0.0, abs=1e-9)
    # One degree of latitude is ~111 km.
    assert haversine_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.19, abs=0.5)


def test_rank_by_distance_orders_nearest_first_and_truncates() -> None:
    rows = [
        _row("Far", 41.0, -73.0),
        _row("Near", 40.7585, -73.9860),
        _row("Mid", 40.80, -73.95),
        _row("NoCoords", None, None),  # type: ignore[arg-type]
        _row("Long", 40.759, -73.986, description="x" * 300),
    ]

    ranked = _rank_by_distance(rows, ORIGIN["lat"], ORIGIN["lng"], limit=3)

    assert [a["name"] for a in ranked][0] in {"Near", "Long"}
    distances = [a["distance_km"] for a in ranked]
    assert distances == sorted(distances)
    assert len(ranked) == 3  # NoCoords row dropped
    long_item = next(a for a in ranked if a["name"] == "Long")
    assert long_item["description"].endswith("…")


@pytest.mark.asyncio
async def test_tool_uses_device_location_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeTool:
        def __init__(self) -> None:
            self.calls: list[tuple[float, float]] = []

        async def nearest(self, lat: float, lng: float, limit: int = 5) -> ToolResponse:
            self.calls.append((lat, lng))
            return ToolResponse(
                status=ToolStatus.SUCCESS,
                summary="Found the 1 nearest attractions.",
                data={"attractions": [{"name": "Near"}]},
            )

    fake = _FakeTool()
    monkeypatch.setattr(attractions_module, "get_attractions_tool", lambda: fake)

    raw = await get_nearest_attractions.ainvoke(
        {},
        config={"configurable": {"lat": ORIGIN["lat"], "lng": ORIGIN["lng"]}},
    )
    result = ToolResponse.model_validate_json(raw)

    assert result.status == ToolStatus.SUCCESS
    assert fake.calls == [(ORIGIN["lat"], ORIGIN["lng"])]


@pytest.mark.asyncio
async def test_tool_warns_without_location(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    class _FakeTool:
        async def nearest(self, lat: float, lng: float, limit: int = 5) -> ToolResponse:
            nonlocal called
            called = True
            return ToolResponse(status=ToolStatus.SUCCESS, summary="", data={})

    monkeypatch.setattr(attractions_module, "get_attractions_tool", lambda: _FakeTool())

    raw = await get_nearest_attractions.ainvoke(
        {},
        config={"configurable": {"user_id": "u1"}},
    )
    result = ToolResponse.model_validate_json(raw)

    assert result.status == ToolStatus.WARNING
    assert called is False


@pytest.mark.asyncio
async def test_nearest_result_excludes_raw_user_coordinates() -> None:
    class _StubTool(AttractionsTool):
        async def _get_client(self) -> object:  # type: ignore[override]
            return object()  # non-None so nearest proceeds

        async def _fetch_rows(self, client: object) -> list[dict[str, Any]]:  # type: ignore[override]
            return [_row("Near", 40.7585, -73.9860), _row("Far", 41.0, -73.0)]

    tool = _StubTool(Settings(_env_file=None))  # type: ignore[call-arg]

    result = await tool.nearest(ORIGIN["lat"], ORIGIN["lng"])
    dumped = result.model_dump_json()

    assert result.status == ToolStatus.SUCCESS
    assert result.data["attractions"]
    # The user's exact location must never reach the model-facing payload.
    assert "origin" not in result.data
    assert "lat" not in dumped and "lng" not in dumped
    for attraction in result.data["attractions"]:
        assert "lat" not in attraction and "lng" not in attraction


@pytest.mark.asyncio
async def test_nearest_warns_when_supabase_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    tool = AttractionsTool(Settings(_env_file=None))  # type: ignore[call-arg]

    result = await tool.nearest(ORIGIN["lat"], ORIGIN["lng"])

    assert result.status == ToolStatus.WARNING
