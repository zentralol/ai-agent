"""Controlled user preference lookup tool."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import lru_cache
from time import perf_counter
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from postgrest.exceptions import APIError
from supabase import AsyncClient, AsyncClientOptions, acreate_client

from app.config import Settings, get_settings
from app.schemas.preferences import UserPreferences
from app.schemas.tools import ToolResponse, ToolStatus

GET_USER_PREFERENCES_TOOL_NAME = "get_user_preferences"
logger = logging.getLogger("zentra_agent.tools.preferences")


@tool
async def get_user_preferences(config: RunnableConfig) -> str:
    """Load compact, sanitized user preferences when personalization is needed."""

    request_id = _configurable_string(config, "request_id")
    conversation_id = _configurable_string(config, "conversation_id")
    user_id = _configurable_string(config, "user_id")
    started_at = perf_counter()

    logger.info(
        "tool_call_start tool=%s request_id=%s conversation_id=%s user=%s",
        GET_USER_PREFERENCES_TOOL_NAME,
        request_id,
        conversation_id,
        _masked_identifier(user_id),
    )
    if user_id is None:
        result = ToolResponse(
            status=ToolStatus.ERROR,
            summary="User preferences cannot be loaded without authenticated user context.",
            next_actions=["Continue without stored preferences for this response."],
        )
        _log_tool_result(
            result=result,
            started_at=started_at,
            request_id=request_id,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        return _tool_response_content(result)

    result = await get_user_preference_tool().get_user_preferences(
        user_id=user_id,
    )
    _log_tool_result(
        result=result,
        started_at=started_at,
        request_id=request_id,
        conversation_id=conversation_id,
        user_id=user_id,
    )
    return _tool_response_content(result)


class UserPreferenceTool:
    """Fetch sanitized user preferences from Supabase using server credentials."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncClient | None = None

    async def get_user_preferences(self, user_id: str) -> ToolResponse:
        """Return a tool envelope containing all stored preferences for one user."""

        if (
            self._settings.supabase_url is None
            or self._settings.supabase_service_role_key is None
        ):
            logger.warning(
                "preferences_tool_supabase_unconfigured table=%s user=%s",
                self._settings.supabase_user_preferences_table,
                _masked_identifier(user_id),
            )
            return ToolResponse(
                status=ToolStatus.WARNING,
                summary="User preferences are unavailable because Supabase is not configured.",
                next_actions=["Continue with neutral defaults or ask a clarifying question."],
            )

        try:
            row = await self._fetch_preference_row(user_id)
        except (APIError, httpx.HTTPError) as exc:
            logger.exception(
                "preferences_tool_supabase_query_failed table=%s user=%s error_type=%s",
                self._settings.supabase_user_preferences_table,
                _masked_identifier(user_id),
                type(exc).__name__,
            )
            return ToolResponse(
                status=ToolStatus.ERROR,
                summary="Failed to load user preferences from Supabase.",
                next_actions=["Continue without stored preferences for this response."],
            )

        if row is None:
            return ToolResponse(
                status=ToolStatus.SUCCESS,
                summary="No stored user preferences were found.",
                data={
                    "preferences": {},
                    "source": "supabase",
                },
            )

        preferences = _preferences_from_row(row)
        return ToolResponse(
            status=ToolStatus.SUCCESS,
            summary="Loaded user preferences.",
            data={
                "preferences": preferences.model_dump(mode="json"),
                "source": "supabase",
            },
        )

    async def _fetch_preference_row(self, user_id: str) -> Mapping[str, Any] | None:
        client = await self._get_client()
        if client is None:
            return None

        started_at = perf_counter()
        response = await (
            client.table(self._settings.supabase_user_preferences_table)
            .select("*")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        if response is None:
            return None
        payload = response.data

        if payload is None:
            return None
        if isinstance(payload, Mapping):
            return payload
        logger.warning(
            "preferences_tool_supabase_unexpected_payload table=%s payload_type=%s "
            "duration_ms=%.2f",
            self._settings.supabase_user_preferences_table,
            type(payload).__name__,
            _duration_ms(started_at),
        )
        return None

    async def _get_client(self) -> AsyncClient | None:
        supabase_url = self._settings.supabase_url
        service_role_key = self._settings.supabase_service_role_key
        if supabase_url is None or service_role_key is None:
            return None
        if self._client is None:
            self._client = await acreate_client(
                supabase_url=supabase_url,
                supabase_key=service_role_key,
                options=AsyncClientOptions(
                    postgrest_client_timeout=self._settings.supabase_timeout_seconds
                ),
            )
        return self._client


@lru_cache(maxsize=1)
def get_user_preference_tool() -> UserPreferenceTool:
    """Return the cached server-side preference lookup tool."""

    return UserPreferenceTool(get_settings())


def _configurable_string(config: RunnableConfig, key: str) -> str | None:
    configurable = config.get("configurable", {})
    value = configurable.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _tool_response_content(result: ToolResponse) -> str:
    return result.model_dump_json()


def _log_tool_result(
    result: ToolResponse,
    started_at: float,
    request_id: str | None,
    conversation_id: str | None,
    user_id: str | None,
) -> None:
    preferences = result.data.get("preferences")
    preference_keys = (
        sorted(preferences)
        if isinstance(preferences, Mapping) and preferences
        else []
    )
    logger.info(
        "tool_call_end tool=%s status=%s request_id=%s conversation_id=%s "
        "user=%s duration_ms=%.2f summary=%r preference_keys=%s",
        GET_USER_PREFERENCES_TOOL_NAME,
        result.status.value,
        request_id,
        conversation_id,
        _masked_identifier(user_id),
        _duration_ms(started_at),
        result.summary,
        preference_keys,
    )


def _duration_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000


def _masked_identifier(value: str | None) -> str:
    if value is None:
        return "<missing>"
    if len(value) <= 8:
        return f"{value[:2]}***"
    return f"{value[:4]}...{value[-4:]}"


def _preferences_from_row(row: Mapping[str, Any]) -> UserPreferences:
    return UserPreferences(
        travel_pace=_optional_string(row.get("travel_pace")),
        crowd_tolerance=_optional_string(row.get("crowd_tolerance")),
        budget_range=_optional_string(row.get("budget_range")),
        interests=_string_list(row.get("interests")),
        mobility_needs=_string_list(row.get("mobility_needs")),
        dietary_needs=_string_list(row.get("dietary_needs")),
        inclusion_needs=_string_list(row.get("inclusion_needs")),
        onboarding_completed=_optional_bool(row.get("onboarding_completed")),
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


def _optional_bool(value: object) -> bool:
    return value if isinstance(value, bool) else False
