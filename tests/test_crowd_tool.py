"""Tests for the crowd-prediction tool."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
import pytest

import app.tools.crowd as crowd_module
from app.config import Settings
from app.schemas.tools import ToolResponse, ToolStatus
from app.tools.crowd import (
    PREDICTIONS_PATH,
    CrowdTool,
    get_crowd_tool,
    predict_crowd_level,
)

ORIGIN = {"lat": 40.7580, "lng": -73.9855}
ISO_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")

_PREDICTION_OK: dict[str, Any] = {
    "success": True,
    "data": {
        "prediction": {
            "busynessScore": 72,
            "busynessLevel": "busy",
            "period": "PM",
            "confidence": 0.8,
            "crowdCategory": "high",
            "modelVersion": "ml-fastapi-v1.0",
        }
    },
    "meta": {"modelVersion": "ml-fastapi-v1.0"},
}


def _settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        BACKEND_API_BASE_URL="http://backend.test",
        AGENT_INTERNAL_TOKEN="svc-token",
    )


def _mock_tool(handler: Any) -> CrowdTool:
    tool = CrowdTool(_settings())
    tool._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return tool


def test_tool_exposes_only_target_time_argument() -> None:
    # target_time is model-supplied; config (and lat/lng) is injected, not an arg.
    assert set(predict_crowd_level.args) == {"target_time"}


@pytest.mark.asyncio
async def test_predict_sends_service_token_and_maps_result() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json=_PREDICTION_OK)

    tool = _mock_tool(handler)
    result = await tool.predict(ORIGIN["lat"], ORIGIN["lng"], "2026-07-10T20:00:00")

    request = captured["request"]
    assert str(request.url) == f"http://backend.test{PREDICTIONS_PATH}"
    assert request.headers["X-Internal-Service-Token"] == "svc-token"
    body = json.loads(request.content)
    assert body["lat"] == ORIGIN["lat"]
    assert body["lng"] == ORIGIN["lng"]
    assert body["targetTime"] == "2026-07-10T20:00:00"
    assert body["durationMinutes"] == 60

    assert result.status == ToolStatus.SUCCESS
    assert result.data["busyness_level"] == "busy"
    assert result.data["busyness_score"] == 72
    # The user's coordinates must not leak into the model-facing payload.
    dumped = result.model_dump_json()
    assert "40.758" not in dumped and "-73.9855" not in dumped


@pytest.mark.asyncio
async def test_predict_out_of_coverage_warns() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={"success": False, "error": {"code": "LOCATION_OUT_OF_COVERAGE"}},
        )

    result = await _mock_tool(handler).predict(0.0, 0.0, "2026-07-10T20:00:00")

    assert result.status == ToolStatus.WARNING
    assert "Manhattan" in result.summary


@pytest.mark.asyncio
async def test_predict_unavailable_warns() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={"success": False, "error": {"code": "PREDICTION_UNAVAILABLE"}},
        )

    result = await _mock_tool(handler).predict(ORIGIN["lat"], ORIGIN["lng"], "t")

    assert result.status == ToolStatus.WARNING


@pytest.mark.asyncio
async def test_predict_errors_on_unexpected_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"success": False, "error": {"code": "X"}})

    result = await _mock_tool(handler).predict(ORIGIN["lat"], ORIGIN["lng"], "t")

    assert result.status == ToolStatus.ERROR


@pytest.mark.asyncio
async def test_predict_warns_without_backend_url() -> None:
    tool = CrowdTool(Settings(_env_file=None))  # type: ignore[call-arg]

    result = await tool.predict(ORIGIN["lat"], ORIGIN["lng"], "t")

    assert result.status == ToolStatus.WARNING


@pytest.mark.asyncio
async def test_tool_passes_location_and_defaults_time_to_now(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeTool:
        def __init__(self) -> None:
            self.calls: list[tuple[float, float, str]] = []

        async def predict(self, lat: float, lng: float, target_time: str) -> ToolResponse:
            self.calls.append((lat, lng, target_time))
            return ToolResponse(status=ToolStatus.SUCCESS, summary="ok", data={})

    fake = _FakeTool()
    monkeypatch.setattr(crowd_module, "get_crowd_tool", lambda: fake)

    raw = await predict_crowd_level.ainvoke(
        {},
        config={"configurable": {"lat": ORIGIN["lat"], "lng": ORIGIN["lng"]}},
    )
    result = ToolResponse.model_validate_json(raw)

    assert result.status == ToolStatus.SUCCESS
    assert len(fake.calls) == 1
    lat, lng, target_time = fake.calls[0]
    assert (lat, lng) == (ORIGIN["lat"], ORIGIN["lng"])
    assert ISO_PATTERN.match(target_time)  # resolved to "now" in NY time


@pytest.mark.asyncio
async def test_tool_warns_without_location(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    class _FakeTool:
        async def predict(self, lat: float, lng: float, target_time: str) -> ToolResponse:
            nonlocal called
            called = True
            return ToolResponse(status=ToolStatus.SUCCESS, summary="", data={})

    monkeypatch.setattr(crowd_module, "get_crowd_tool", lambda: _FakeTool())

    raw = await predict_crowd_level.ainvoke(
        {},
        config={"configurable": {"user_id": "u1"}},
    )
    result = ToolResponse.model_validate_json(raw)

    assert result.status == ToolStatus.WARNING
    assert called is False


def test_get_crowd_tool_is_cached() -> None:
    assert get_crowd_tool() is get_crowd_tool()
