"""Itinerary planning tool — calls zentra-recommend via the Express gateway.

Fetches the user's stored preferences from Supabase, maps them to the
inline_profile format zentra-recommend expects, then POSTs to
/itinerary/plan and returns the full day plan.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from time import perf_counter

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.config import Settings, get_settings
from app.schemas.tools import ToolResponse, ToolStatus
from app.tools.preferences import get_user_preference_tool

PLAN_ITINERARY_TOOL_NAME = "plan_itinerary"
ITINERARY_PATH = "/itinerary/plan"
SERVICE_TOKEN_HEADER = "X-Internal-Service-Token"
HTTP_TIMEOUT_SECONDS = 60.0

logger = logging.getLogger("zentra_agent.tools.itinerary")

# Supabase field → inline_profile field
_PREF_MAP = {
    "travel_pace":   "pace",
    "budget_range":  "budget",
    "interests":     "interests",
    "dietary_needs": "dietary_preferences",
    "mobility_needs": "accessibility_needs",
    "inclusion_needs": "avoid",
}


@tool
async def plan_itinerary(
    config: RunnableConfig,
    anchor_place: str,
    anchor_time: str,
    duration_hours: int = 8,
    additional_context: str = "",
) -> str:
    """Build a personalised Manhattan day itinerary for the user.

    Call this when the user asks to plan their day, create an itinerary, or
    suggests a starting place and time. Pass the place name exactly as the
    user said it (e.g. "The High Line", "Central Park"). Pass anchor_time as
    an ISO 8601 date-time in New York local time (e.g. "2026-07-10T10:00:00").
    Pass any extra context the user mentioned (stroller, anniversary, etc.) in
    additional_context. The tool fetches the user's stored preferences and
    returns a full stop-by-stop itinerary with crowd levels and travel tips.
    """

    request_id = _configurable_string(config, "request_id")
    conversation_id = _configurable_string(config, "conversation_id")
    user_id = _configurable_string(config, "user_id")
    started_at = perf_counter()

    logger.info(
        "tool_call_start tool=%s request_id=%s conversation_id=%s anchor_place=%r anchor_time=%s",
        PLAN_ITINERARY_TOOL_NAME,
        request_id,
        conversation_id,
        anchor_place,
        anchor_time,
    )

    result = await get_itinerary_tool().plan(
        user_id=user_id,
        anchor_place=anchor_place,
        anchor_time=anchor_time,
        duration_hours=duration_hours,
        additional_context=additional_context,
    )

    logger.info(
        "tool_call_end tool=%s status=%s request_id=%s conversation_id=%s duration_ms=%.2f",
        PLAN_ITINERARY_TOOL_NAME,
        result.status.value,
        request_id,
        conversation_id,
        _duration_ms(started_at),
    )
    return result.model_dump_json()


class ItineraryTool:
    """Fetch preferences then call zentra-recommend to build a day plan."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    async def plan(
        self,
        user_id: str | None,
        anchor_place: str,
        anchor_time: str,
        duration_hours: int,
        additional_context: str,
    ) -> ToolResponse:
        base_url = (self._settings.backend_api_base_url or "").rstrip("/")
        if not base_url:
            logger.warning("itinerary_tool_unconfigured missing=BACKEND_API_BASE_URL")
            return ToolResponse(
                status=ToolStatus.WARNING,
                summary="Itinerary planning is unavailable; the backend is not configured.",
                next_actions=["Check that BACKEND_API_BASE_URL is set."],
            )

        inline_profile = await self._build_inline_profile(user_id)

        payload = {
            "inline_profile": inline_profile,
            "anchor_place": anchor_place,
            "anchor_time": anchor_time,
            "duration_hours": duration_hours,
            "additional_context": additional_context,
        }

        try:
            response = await self._post(base_url, payload)
        except httpx.HTTPError as exc:
            logger.exception("itinerary_tool_request_failed error_type=%s", type(exc).__name__)
            return ToolResponse(
                status=ToolStatus.ERROR,
                summary="Failed to reach the itinerary planning service.",
                next_actions=["Try again or suggest the user retry shortly."],
            )

        return _interpret_response(response)

    async def _build_inline_profile(self, user_id: str | None) -> dict:
        """Fetch stored preferences and map to inline_profile. Falls back to defaults."""

        defaults: dict = {
            "name": "Traveller",
            "interests": [],
            "dietary_preferences": [],
            "accessibility_needs": [],
            "budget": "moderate",
            "pace": "moderate",
            "travel_with": "solo",
            "avoid": [],
        }

        if user_id is None:
            return defaults

        try:
            pref_result = await get_user_preference_tool().get_user_preferences(user_id)
        except Exception:
            logger.warning(
                "itinerary_tool_preference_fetch_failed user=%s",
                user_id[:4] if user_id else "?",
            )
            return defaults

        prefs: dict = pref_result.data.get("preferences", {})
        if not prefs:
            return defaults

        profile = dict(defaults)
        for supabase_field, profile_field in _PREF_MAP.items():
            value = prefs.get(supabase_field)
            if value is not None and value != "" and value != []:
                profile[profile_field] = value

        return profile

    async def _post(self, base_url: str, payload: dict) -> httpx.Response:
        headers = {"Content-Type": "application/json"}
        token = self._settings.agent_internal_token
        if token:
            headers[SERVICE_TOKEN_HEADER] = token

        started_at = perf_counter()
        response = await self._get_client().post(
            f"{base_url}{ITINERARY_PATH}",
            headers=headers,
            json=payload,
        )
        logger.info(
            "itinerary_tool_request_end status=%d duration_ms=%.2f",
            response.status_code,
            _duration_ms(started_at),
        )
        return response

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS)
        return self._client


@lru_cache(maxsize=1)
def get_itinerary_tool() -> ItineraryTool:
    """Return the cached itinerary tool."""

    return ItineraryTool(get_settings())


def _interpret_response(response: httpx.Response) -> ToolResponse:
    if response.status_code != 200:
        logger.warning("itinerary_tool_bad_status status=%d", response.status_code)
        try:
            detail = response.json().get("detail", "unknown error")
        except Exception:
            detail = response.text[:200]
        return ToolResponse(
            status=ToolStatus.ERROR,
            summary=f"Itinerary service returned {response.status_code}: {detail}",
            next_actions=["Inform the user the plan could not be built and to try again."],
        )

    try:
        data = response.json()
    except Exception:
        return ToolResponse(
            status=ToolStatus.ERROR,
            summary="Itinerary service returned an unreadable response.",
            next_actions=["Try again or suggest a simpler request."],
        )

    stops = data.get("stops", [])
    start_time = stops[0].get("time", "?") if stops else "?"
    return ToolResponse(
        status=ToolStatus.SUCCESS,
        summary=f"Itinerary built: {len(stops)} stops starting at {start_time}.",
        data=data,
    )


def _configurable_string(config: RunnableConfig, key: str) -> str | None:
    value = config.get("configurable", {}).get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _duration_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000
