"""Tests for the controlled user preference lookup tool."""

from __future__ import annotations

import pytest

import app.tools.preferences as preference_tools
from app.config import Settings
from app.schemas.preferences import PreferenceCategory
from app.schemas.tools import ToolResponse, ToolStatus
from app.tools.preferences import (
    UserPreferenceTool,
    get_user_preferences,
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


def test_tool_schema_hides_injected_config_and_user_id() -> None:
    assert "categories" in get_user_preferences.args
    assert "config" not in get_user_preferences.args
    assert "user_id" not in get_user_preferences.args


def test_parse_preference_categories_keeps_only_allowed_values() -> None:
    categories = parse_preference_categories(["crowd", "bad-value", "transport", 123])

    assert categories == (PreferenceCategory.CROWD, PreferenceCategory.TRANSPORT)


@pytest.mark.asyncio
async def test_langchain_tool_uses_runtime_config_for_user_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePreferenceTool:
        async def get_user_preferences(
            self, user_id: str, categories: tuple[PreferenceCategory, ...]
        ) -> ToolResponse:
            assert user_id == "u1"
            assert categories == (PreferenceCategory.CROWD,)
            return ToolResponse(
                status=ToolStatus.SUCCESS,
                summary="Loaded user preferences.",
                data={"preferences": {"crowd_tolerance": "low"}},
            )

    monkeypatch.setattr(
        preference_tools,
        "get_user_preference_tool",
        lambda: FakePreferenceTool(),
    )

    raw_result = await get_user_preferences.ainvoke(
        {"categories": ["crowd"]},
        config={"configurable": {"user_id": "u1"}},
    )
    result = ToolResponse.model_validate_json(raw_result)

    assert result.status == ToolStatus.SUCCESS
    assert result.data["preferences"] == {"crowd_tolerance": "low"}


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
