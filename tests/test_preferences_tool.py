"""Tests for the controlled user preference lookup tool."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.schemas.preferences import PreferenceCategory
from app.schemas.tools import ToolStatus
from app.tools.preferences import (
    GET_USER_PREFERENCES_TOOL_SCHEMA,
    UserPreferenceTool,
    parse_preference_categories,
)


class _FakeRowPreferenceTool(UserPreferenceTool):
    async def _fetch_preference_row(self, user_id: str) -> dict[str, object] | None:
        assert user_id == "u1"
        return {
            "crowd_tolerance": "low",
            "preferred_transport": "walk",
            "preferences": {
                "language": "zh",
                "interests": ["parks", "museums"],
            },
        }


def test_tool_schema_does_not_accept_model_supplied_user_id() -> None:
    function = GET_USER_PREFERENCES_TOOL_SCHEMA["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)

    assert "categories" in properties
    assert "user_id" not in properties


def test_parse_preference_categories_keeps_only_allowed_values() -> None:
    categories = parse_preference_categories(["crowd", "bad-value", "transport", 123])

    assert categories == (PreferenceCategory.CROWD, PreferenceCategory.TRANSPORT)


@pytest.mark.asyncio
async def test_preference_tool_warns_when_supabase_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    tool = UserPreferenceTool(settings)

    result = await tool.get_user_preferences("u1", (PreferenceCategory.CROWD,))

    assert result.status == ToolStatus.WARNING
    assert result.data == {"categories": ["crowd"]}


@pytest.mark.asyncio
async def test_preference_tool_returns_only_requested_categories() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        SUPABASE_URL="https://example.supabase.co",
        SUPABASE_SERVICE_ROLE_KEY="secret",
    )
    tool = _FakeRowPreferenceTool(settings)

    result = await tool.get_user_preferences(
        "u1",
        (PreferenceCategory.CROWD, PreferenceCategory.LANGUAGE),
    )

    assert result.status == ToolStatus.SUCCESS
    assert result.data["preferences"] == {
        "crowd_tolerance": "low",
        "language": "zh",
    }
