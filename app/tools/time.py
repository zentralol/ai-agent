"""Current date and time tool for New York local scheduling."""

from __future__ import annotations

import logging
from datetime import datetime
from time import perf_counter
from zoneinfo import ZoneInfo

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.schemas.tools import ToolResponse, ToolStatus

GET_CURRENT_TIME_TOOL_NAME = "get_current_time"
NEW_YORK_TZ = ZoneInfo("America/New_York")

logger = logging.getLogger("zentra_agent.tools.time")


def current_time_payload(now: datetime | None = None) -> dict[str, str]:
    """Build the time payload for the given moment in New York local time."""

    moment = (now or datetime.now(NEW_YORK_TZ)).astimezone(NEW_YORK_TZ)
    return {
        "timezone": "America/New_York",
        "iso_datetime": moment.strftime("%Y-%m-%dT%H:%M:%S"),
        "date": moment.strftime("%Y-%m-%d"),
        "time": moment.strftime("%H:%M:%S"),
        "day_of_week": moment.strftime("%A"),
    }


@tool
async def get_current_time(config: RunnableConfig) -> str:
    """Get the current date and time in New York (America/New_York).

    Call this when you need today's date, the current time, or to compute
    relative times like "tomorrow" or "this afternoon" before calling
    plan_itinerary or predict_crowd_level. Do not ask the user what day it is.
    """

    request_id = _configurable_string(config, "request_id")
    conversation_id = _configurable_string(config, "conversation_id")
    started_at = perf_counter()

    logger.info(
        "tool_call_start tool=%s request_id=%s conversation_id=%s",
        GET_CURRENT_TIME_TOOL_NAME,
        request_id,
        conversation_id,
    )

    data = current_time_payload()
    result = ToolResponse(
        status=ToolStatus.SUCCESS,
        summary=(
            f"Current New York time is {data['iso_datetime']} ({data['day_of_week']})."
        ),
        data=data,
    )

    logger.info(
        "tool_call_end tool=%s status=%s request_id=%s conversation_id=%s duration_ms=%.2f",
        GET_CURRENT_TIME_TOOL_NAME,
        result.status.value,
        request_id,
        conversation_id,
        _duration_ms(started_at),
    )
    return result.model_dump_json()


def _configurable_string(config: RunnableConfig, key: str) -> str | None:
    value = config.get("configurable", {}).get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _duration_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000
