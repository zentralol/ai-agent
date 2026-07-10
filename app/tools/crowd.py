"""Crowd-density prediction tool backed by the Express backend.

Calls the backend ``/api/v1/predictions`` capability (which the agent is now
authorized to reach with the internal service token) to tell the user how busy
their current location is, or will be at a given time. Coverage is Manhattan
only; the user's coordinates come from the device and are never returned.
"""

from __future__ import annotations

import logging
from datetime import datetime
from functools import lru_cache
from time import perf_counter
from zoneinfo import ZoneInfo

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.config import Settings, get_settings
from app.geo import configurable_float
from app.schemas.tools import ToolResponse, ToolStatus

GET_CROWD_LEVEL_TOOL_NAME = "predict_crowd_level"

PREDICTIONS_PATH = "/api/v1/predictions"
SERVICE_TOKEN_HEADER = "X-Internal-Service-Token"
NEW_YORK_TZ = ZoneInfo("America/New_York")
DURATION_MINUTES = 60
HTTP_TIMEOUT_SECONDS = 8.0

logger = logging.getLogger("zentra_agent.tools.crowd")


@tool
async def predict_crowd_level(config: RunnableConfig, target_time: str = "") -> str:
    """Predict how crowded the user's current location is (Manhattan only).

    Use this when the user asks how busy or crowded it is near them. Optionally
    pass ``target_time`` as an ISO 8601 date-time (New York local time, e.g.
    "2026-07-10T20:00:00") to forecast a specific time; omit it for right now.
    The location comes from the user's device, not from you.

    When you answer, describe the crowd level in one short, natural sentence
    (e.g. "It's fairly quiet near you right now."). Do NOT show the user raw
    fields such as confidence, period, or model version — they are internal
    metadata, not part of the reply.
    """

    request_id = _configurable_string(config, "request_id")
    conversation_id = _configurable_string(config, "conversation_id")
    lat = configurable_float(config, "lat")
    lng = configurable_float(config, "lng")
    resolved_time = _resolve_target_time(target_time)
    started_at = perf_counter()

    logger.info(
        "tool_call_start tool=%s request_id=%s conversation_id=%s has_location=%s target_time=%s",
        GET_CROWD_LEVEL_TOOL_NAME,
        request_id,
        conversation_id,
        lat is not None and lng is not None,
        resolved_time,
    )

    if lat is None or lng is None:
        result = ToolResponse(
            status=ToolStatus.WARNING,
            summary="The user's current location is not available.",
            next_actions=[
                "Ask the user to share their location or name a place to check."
            ],
        )
    else:
        result = await get_crowd_tool().predict(lat, lng, resolved_time)

    logger.info(
        "tool_call_end tool=%s status=%s request_id=%s conversation_id=%s duration_ms=%.2f",
        GET_CROWD_LEVEL_TOOL_NAME,
        result.status.value,
        request_id,
        conversation_id,
        _duration_ms(started_at),
    )
    return result.model_dump_json()


class CrowdTool:
    """Call the backend crowd-prediction endpoint with the service token."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    async def predict(self, lat: float, lng: float, target_time: str) -> ToolResponse:
        """Return a busyness prediction for ``(lat, lng)`` at ``target_time``."""

        base_url = (self._settings.backend_api_base_url or "").rstrip("/")
        if not base_url:
            logger.warning("crowd_tool_unconfigured missing=BACKEND_API_BASE_URL")
            return ToolResponse(
                status=ToolStatus.WARNING,
                summary="Crowd prediction is unavailable; the backend is not configured.",
                next_actions=["Continue without crowd data or ask a clarifying question."],
            )

        try:
            response = await self._post(base_url, lat, lng, target_time)
        except httpx.HTTPError as exc:
            logger.exception("crowd_tool_request_failed error_type=%s", type(exc).__name__)
            return ToolResponse(
                status=ToolStatus.ERROR,
                summary="Failed to reach the crowd-prediction service.",
                next_actions=["Continue without crowd data for this response."],
            )

        return _interpret_response(response, target_time)

    async def _post(
        self, base_url: str, lat: float, lng: float, target_time: str
    ) -> httpx.Response:
        headers = {"Content-Type": "application/json"}
        token = self._settings.agent_internal_token
        if token:
            headers[SERVICE_TOKEN_HEADER] = token

        response = await self._get_client().post(
            f"{base_url}{PREDICTIONS_PATH}",
            headers=headers,
            json={
                "lat": lat,
                "lng": lng,
                "targetTime": target_time,
                "durationMinutes": DURATION_MINUTES,
            },
        )
        return response

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS)
        return self._client


@lru_cache(maxsize=1)
def get_crowd_tool() -> CrowdTool:
    """Return the cached crowd-prediction tool."""

    return CrowdTool(get_settings())


def _interpret_response(response: httpx.Response, target_time: str) -> ToolResponse:
    if response.status_code == 200:
        return _success_response(response, target_time)

    code = _error_code(response)
    if code == "LOCATION_OUT_OF_COVERAGE":
        return ToolResponse(
            status=ToolStatus.WARNING,
            summary="Crowd prediction is only available for Manhattan.",
            next_actions=["Tell the user this feature currently covers Manhattan only."],
        )
    if response.status_code == 503 or code == "PREDICTION_UNAVAILABLE":
        return ToolResponse(
            status=ToolStatus.WARNING,
            summary="No crowd prediction is available for this location and time.",
            next_actions=["Continue without crowd data or suggest another spot."],
        )
    logger.warning("crowd_tool_bad_status status=%d code=%s", response.status_code, code)
    return ToolResponse(
        status=ToolStatus.ERROR,
        summary="The crowd-prediction service returned an error.",
        next_actions=["Continue without crowd data for this response."],
    )


def _success_response(response: httpx.Response, target_time: str) -> ToolResponse:
    payload = response.json() if response.content else {}
    prediction = (
        payload.get("data", {}).get("prediction")
        if isinstance(payload, dict)
        else None
    )
    if not isinstance(prediction, dict):
        return ToolResponse(
            status=ToolStatus.WARNING,
            summary="No crowd prediction is available for this location and time.",
        )

    level = prediction.get("busynessLevel")
    score = prediction.get("busynessScore")
    return ToolResponse(
        status=ToolStatus.SUCCESS,
        summary=f"Crowd level near the user: {level} (score {score}).",
        data={
            "busyness_level": level,
            "busyness_score": score,
            "period": prediction.get("period"),
            "confidence": prediction.get("confidence"),
            "crowd_category": prediction.get("crowdCategory"),
            "model_version": prediction.get("modelVersion"),
            "target_time": target_time,
        },
    )


def _error_code(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        code = payload["error"].get("code")
        return code if isinstance(code, str) else None
    return None


def _resolve_target_time(target_time: str) -> str:
    stripped = target_time.strip()
    if stripped:
        return stripped
    return datetime.now(NEW_YORK_TZ).strftime("%Y-%m-%dT%H:%M:%S")


def _configurable_string(config: RunnableConfig, key: str) -> str | None:
    value = config.get("configurable", {}).get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _duration_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000
