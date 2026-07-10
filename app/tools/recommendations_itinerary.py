"""Recommendation tool — fetches personalised place cards from zentra-recommend.

Returns place cards with images, descriptions, crowd levels, and highlights.
Use this when the user asks for suggestions, recommendations, or "what should
I visit" without wanting a full day itinerary.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from time import perf_counter
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.config import Settings, get_settings
from app.schemas.tools import ToolResponse, ToolStatus
from app.tools.preferences import get_user_preference_tool

RECOMMEND_TOOL_NAME = "get_place_recommendations"
RECOMMEND_PATH = "/api/v1/recommend"
SERVICE_TOKEN_HEADER = "X-Internal-Service-Token"
HTTP_TIMEOUT_SECONDS = 20.0

logger = logging.getLogger("zentra_agent.tools.recommendations_itinerary")

_PREF_MAP = {
    "travel_pace":    "pace",
    "budget_range":   "budget",
    "interests":      "interests",
    "dietary_needs":  "dietary_preferences",
    "mobility_needs": "accessibility_needs",
    "inclusion_needs": "avoid",
}


@tool
async def get_place_recommendations(
    config: RunnableConfig,
    query: str = "",
    category: str = "",
    budget: str = "",
    count: int = 6,
) -> str:
    """Get personalised Manhattan place recommendations with images and crowd levels.

    Use this when the user asks for suggestions or "what should I visit" without
    wanting a full planned itinerary. Pass a natural language query describing
    what they want (e.g. "quiet art museums", "free outdoor parks for families").
    Optionally filter by category (park, museum, food, landmark, entertainment,
    shopping, bar, art, neighborhood, sports) or budget (free, budget, moderate,
    luxury). count controls how many results to return (default 6, max 12).

    When presenting results, show the name, neighborhood, description, and crowd
    level for each place. Mention the image only if the user asks for visuals.
    Do not show raw fields like crowd_score or lat/lon.
    """

    request_id = _configurable_string(config, "request_id")
    user_id = _configurable_string(config, "user_id")
    started_at = perf_counter()

    logger.info(
        "tool_call_start tool=%s request_id=%s query=%r category=%s",
        RECOMMEND_TOOL_NAME, request_id, query, category,
    )

    result = await get_recommendations_tool().recommend(
        user_id=user_id,
        query=query,
        category=category or None,
        budget=budget or None,
        count=count,
    )

    logger.info(
        "tool_call_end tool=%s status=%s request_id=%s duration_ms=%.2f",
        RECOMMEND_TOOL_NAME, result.status.value, request_id, _duration_ms(started_at),
    )
    return result.model_dump_json()


class RecommendationsTool:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    async def recommend(
        self,
        user_id: str | None,
        query: str,
        category: str | None,
        budget: str | None,
        count: int,
    ) -> ToolResponse:
        base_url = (self._settings.backend_api_base_url or "").rstrip("/")
        if not base_url:
            logger.warning("recommendations_tool_unconfigured missing=BACKEND_API_BASE_URL")
            return ToolResponse(
                status=ToolStatus.WARNING,
                summary="Recommendations are unavailable; the backend is not configured.",
                next_actions=["Check that BACKEND_API_BASE_URL is set."],
            )

        inline_profile = await self._build_inline_profile(user_id)

        payload: dict[str, Any] = {
            "inline_profile": inline_profile,
            "count": count,
        }
        if query:
            payload["query"] = query
        if category:
            payload["category"] = category
        if budget:
            payload["budget"] = budget

        try:
            response = await self._post(base_url, payload)
        except httpx.HTTPError as exc:
            logger.exception(
                "recommendations_tool_request_failed error_type=%s", type(exc).__name__
            )
            return ToolResponse(
                status=ToolStatus.ERROR,
                summary="Failed to reach the recommendations service.",
                next_actions=["Try again or suggest a specific place instead."],
            )

        return _interpret_response(response)

    async def _build_inline_profile(self, user_id: str | None) -> dict[str, Any]:
        defaults: dict[str, Any] = {
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
            prefs: dict[str, Any] = pref_result.data.get("preferences", {})
            if not prefs:
                return defaults
            profile = dict(defaults)
            for supabase_field, profile_field in _PREF_MAP.items():
                value = prefs.get(supabase_field)
                if value is not None and value != "" and value != []:
                    profile[profile_field] = value
            return profile
        except Exception:
            return defaults

    async def _post(self, base_url: str, payload: dict[str, Any]) -> httpx.Response:
        headers = {"Content-Type": "application/json"}
        token = self._settings.agent_internal_token
        if token:
            headers[SERVICE_TOKEN_HEADER] = token
        return await self._get_client().post(
            f"{base_url}{RECOMMEND_PATH}",
            headers=headers,
            json=payload,
            timeout=HTTP_TIMEOUT_SECONDS,
        )

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS)
        return self._client


@lru_cache(maxsize=1)
def get_recommendations_tool() -> RecommendationsTool:
    return RecommendationsTool(get_settings())


def _interpret_response(response: httpx.Response) -> ToolResponse:
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", "unknown error")
        except Exception:
            detail = response.text[:200]
        return ToolResponse(
            status=ToolStatus.ERROR,
            summary=f"Recommendations service returned {response.status_code}: {detail}",
            next_actions=["Suggest a specific place or try a broader query."],
        )
    try:
        data = response.json()
    except Exception:
        return ToolResponse(
            status=ToolStatus.ERROR,
            summary="Recommendations service returned an unreadable response.",
        )

    recs = data.get("recommendations", [])
    based_on = data.get("based_on", "")
    return ToolResponse(
        status=ToolStatus.SUCCESS,
        summary=f"{len(recs)} recommendations returned. {based_on}",
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
