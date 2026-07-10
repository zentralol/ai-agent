"""Tests for the current time lookup tool."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.schemas.tools import ToolResponse, ToolStatus
from app.tools.time import NEW_YORK_TZ, current_time_payload, get_current_time

FIXED_NOW = datetime(2026, 7, 10, 15, 4, 0, tzinfo=NEW_YORK_TZ)


def test_tool_schema_hides_injected_config() -> None:
    assert get_current_time.args == {}
    assert "config" not in get_current_time.args


def test_current_time_payload_formats_new_york_time() -> None:
    data = current_time_payload(FIXED_NOW)

    assert data == {
        "timezone": "America/New_York",
        "iso_datetime": "2026-07-10T15:04:00",
        "date": "2026-07-10",
        "time": "15:04:00",
        "day_of_week": "Friday",
    }


def test_current_time_payload_converts_other_timezones() -> None:
    paris_time = datetime(2026, 7, 10, 21, 4, 0, tzinfo=ZoneInfo("Europe/Paris"))
    data = current_time_payload(paris_time)

    assert data["iso_datetime"] == "2026-07-10T15:04:00"
    assert data["day_of_week"] == "Friday"


@pytest.mark.asyncio
async def test_langchain_tool_returns_current_new_york_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: ZoneInfo | None = None) -> datetime:
            assert tz == NEW_YORK_TZ
            return FIXED_NOW

    monkeypatch.setattr("app.tools.time.datetime", _FixedDatetime)

    raw_result = await get_current_time.ainvoke(
        {},
        config={"configurable": {"request_id": "req-1", "conversation_id": "conv-1"}},
    )
    result = ToolResponse.model_validate_json(raw_result)

    assert result.status == ToolStatus.SUCCESS
    assert result.data["iso_datetime"] == "2026-07-10T15:04:00"
    assert result.data["date"] == "2026-07-10"
    assert result.data["day_of_week"] == "Friday"
    assert "Friday" in result.summary
