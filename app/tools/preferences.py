"""Controlled user preference lookup tool.

The model never supplies a user id to this tool. The agent passes the
authenticated user id from the request context and only allows the model or
router to request narrow preference categories.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any
from urllib.parse import quote

import httpx

from app.config import Settings, get_settings
from app.schemas.preferences import (
    PreferenceCategory,
    UserPreferences,
    dump_selected_preferences,
)
from app.schemas.tools import ToolResponse, ToolStatus

PLANNING_PREFERENCE_CATEGORIES: tuple[PreferenceCategory, ...] = (
    PreferenceCategory.TRAVEL_STYLE,
    PreferenceCategory.CROWD,
    PreferenceCategory.TRANSPORT,
    PreferenceCategory.BUDGET,
    PreferenceCategory.ACCESSIBILITY,
    PreferenceCategory.LANGUAGE,
    PreferenceCategory.INTERESTS,
)

_PLANNING_KEYWORDS = (
    "route",
    "itinerary",
    "plan",
    "recommend",
    "recommendation",
    "quiet",
    "crowd",
    "busy",
    "路线",
    "行程",
    "规划",
    "计划",
    "推荐",
    "安静",
    "人少",
    "拥挤",
    "避开",
    "偏好",
    "喜欢",
)


def infer_preference_categories(message: str) -> tuple[PreferenceCategory, ...]:
    """Return preference categories worth loading for the user message."""

    normalized = message.casefold()
    if any(keyword in normalized for keyword in _PLANNING_KEYWORDS):
        return PLANNING_PREFERENCE_CATEGORIES
    return ()


class UserPreferenceTool:
    """Fetch sanitized user preferences from Supabase using server credentials."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def get_user_preferences(
        self, user_id: str, categories: Sequence[PreferenceCategory]
    ) -> ToolResponse:
        """Return a tool envelope containing only requested preference groups."""

        requested_categories = _dedupe_categories(categories)
        if not requested_categories:
            return ToolResponse(
                status=ToolStatus.WARNING,
                summary="No preference categories were requested.",
            )

        if (
            self._settings.supabase_url is None
            or self._settings.supabase_service_role_key is None
        ):
            return ToolResponse(
                status=ToolStatus.WARNING,
                summary="User preferences are unavailable because Supabase is not configured.",
                data={"categories": [category.value for category in requested_categories]},
                next_actions=["Continue with neutral defaults or ask a clarifying question."],
            )

        try:
            row = await self._fetch_preference_row(user_id)
        except httpx.HTTPError:
            return ToolResponse(
                status=ToolStatus.ERROR,
                summary="Failed to load user preferences from Supabase.",
                data={"categories": [category.value for category in requested_categories]},
                next_actions=["Continue without stored preferences for this response."],
            )

        if row is None:
            return ToolResponse(
                status=ToolStatus.SUCCESS,
                summary="No stored user preferences were found.",
                data={
                    "categories": [category.value for category in requested_categories],
                    "preferences": {},
                    "source": "supabase",
                },
            )

        preferences = _preferences_from_row(row)
        selected = dump_selected_preferences(preferences, requested_categories)
        return ToolResponse(
            status=ToolStatus.SUCCESS,
            summary="Loaded user preferences.",
            data={
                "categories": [category.value for category in requested_categories],
                "preferences": selected,
                "source": "supabase",
            },
        )

    async def _fetch_preference_row(self, user_id: str) -> Mapping[str, Any] | None:
        supabase_url = self._settings.supabase_url
        service_role_key = self._settings.supabase_service_role_key
        if supabase_url is None or service_role_key is None:
            return None

        table_name = quote(self._settings.supabase_user_preferences_table, safe="")
        url = f"{supabase_url.rstrip('/')}/rest/v1/{table_name}"
        headers = {
            "apikey": service_role_key,
            "authorization": f"Bearer {service_role_key}",
            "accept": "application/json",
        }
        params = {
            "user_id": f"eq.{user_id}",
            "select": "*",
            "limit": "1",
        }

        async with httpx.AsyncClient(
            timeout=self._settings.supabase_timeout_seconds
        ) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            payload = response.json()

        if not isinstance(payload, list) or not payload:
            return None

        first = payload[0]
        if not isinstance(first, Mapping):
            return None
        return first


@lru_cache(maxsize=1)
def get_user_preference_tool() -> UserPreferenceTool:
    """Return the cached server-side preference lookup tool."""

    return UserPreferenceTool(get_settings())


def _dedupe_categories(
    categories: Sequence[PreferenceCategory],
) -> tuple[PreferenceCategory, ...]:
    return tuple(dict.fromkeys(categories))


def _preferences_from_row(row: Mapping[str, Any]) -> UserPreferences:
    nested_preferences = row.get("preferences")
    raw: dict[str, Any] = {}
    if isinstance(nested_preferences, Mapping):
        raw.update(nested_preferences)
    raw.update({key: row[key] for key in UserPreferences.model_fields if key in row})

    return UserPreferences(
        travel_style=_optional_string(raw.get("travel_style")),
        crowd_tolerance=_optional_string(raw.get("crowd_tolerance")),
        preferred_transport=_optional_string(raw.get("preferred_transport")),
        budget=_optional_string(raw.get("budget")),
        accessibility=_string_list(raw.get("accessibility")),
        language=_optional_string(raw.get("language")),
        interests=_string_list(raw.get("interests")),
    )


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]
