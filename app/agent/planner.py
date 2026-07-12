"""Multi-day trip planner built on serial single-day itinerary calls."""

from __future__ import annotations

import datetime
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from app.agent.trip_state import DayPlan, TripState
from app.schemas.events import (
    DoneEvent,
    MessageDeltaEvent,
    StreamEvent,
    ToolFinishedEvent,
    ToolStartedEvent,
    WarningEvent,
)
from app.schemas.recommendations import RecommendationData, RecommendationItem
from app.schemas.tools import ToolResponse, ToolStatus
from app.tools.itinerary import PLAN_ITINERARY_TOOL_NAME

logger = logging.getLogger("zentra_agent.agent.planner")
def _tool_config_from_request(request: Any) -> RunnableConfig:
    return {
        "configurable": {
            "user_id": getattr(request, "user_id", None),
            "request_id": getattr(request, "request_id", None),
            "conversation_id": getattr(request, "conversation_id", None),
            "lat": getattr(request, "lat", None),
            "lng": getattr(request, "lng", None),
        }
    }


class MultiDayPlanner:
    """Coordinate multiple single-day plan_itinerary calls into one trip."""

    async def plan_multi_day(
        self,
        request: Any,
        state: TripState,
        model: BaseChatModel,
        tools: tuple[BaseTool, ...],
    ) -> tuple[list[StreamEvent], RecommendationData, TripState]:
        """Plan a multi-day trip by calling the day planner once per day."""

        itinerary_tool = _find_tool(tools, PLAN_ITINERARY_TOOL_NAME)
        if itinerary_tool is None:
            raise ValueError(f"Required tool {PLAN_ITINERARY_TOOL_NAME} not found")

        num_days = max(1, min(state.num_days or 1, 14))
        start_date = state.start_date or datetime.date.today().isoformat()
        anchor_place = state.anchor_place or "Manhattan"
        base_context = state.additional_context or ""

        events: list[StreamEvent] = []
        day_plans: dict[str, DayPlan] = {}
        visited_place_ids: list[str] = list(state.visited_place_ids)
        failed_days: list[str] = []

        for offset in range(num_days):
            date = _offset_date(start_date, offset)
            anchor_time = f"{date}T10:00:00"
            theme = _theme_for_day(offset, base_context)
            context = _build_day_context(theme, visited_place_ids, base_context)

            events.append(ToolStartedEvent(tool_name=PLAN_ITINERARY_TOOL_NAME))
            tool_result = await _call_itinerary_tool(
                itinerary_tool, anchor_place, anchor_time, context, request
            )
            events.append(
                ToolFinishedEvent(
                    tool_name=PLAN_ITINERARY_TOOL_NAME,
                    result=tool_result,
                )
            )

            if tool_result.status is not ToolStatus.SUCCESS:
                failed_days.append(date)
                continue

            stops, place_ids = _parse_stops(tool_result.data, visited_place_ids)
            if stops:
                visited_place_ids.extend(place_ids)
                day_plans[date] = DayPlan(
                    date=date,
                    anchor_place=anchor_place,
                    anchor_time=anchor_time,
                    theme=theme,
                    stops=stops,
                )

        merged_items = _merge_day_stops(day_plans)
        if not merged_items:
            events.append(
                WarningEvent(
                    message="I couldn't build any days for this trip. "
                    "Please try again with a different area or dates."
                )
            )
            data = RecommendationData(
                source="itinerary",
                items=[],
                start_date=start_date,
                end_date=_offset_date(start_date, num_days - 1),
            )
        else:
            data = RecommendationData(
                source="itinerary",
                items=merged_items,
                start_date=start_date,
                end_date=_offset_date(start_date, num_days - 1),
                summary=_summarize_plan(start_date, num_days, merged_items),
            )
        updated_state = state.model_copy(
            update={
                "mode": "multi",
                "day_plans": day_plans,
                "visited_place_ids": visited_place_ids,
            }
        )
        return events, data, updated_state

    async def modify_day(
        self,
        request: Any,
        state: TripState,
        model: BaseChatModel,
        tools: tuple[BaseTool, ...],
        target: str,
    ) -> tuple[list[StreamEvent], RecommendationData, TripState]:
        """Replan a single day while preserving the rest of the trip."""

        target_date = _resolve_target_date(target, state)
        if target_date is None or target_date not in state.day_plans:
            events: list[StreamEvent] = [
                MessageDeltaEvent(
                    text=f"I couldn't find day {target} in your current trip."
                ),
                DoneEvent(),
            ]
            return events, RecommendationData(source="itinerary", items=[]), state

        # Rebuild visited_place_ids from all days except the target day.
        visited_place_ids: list[str] = []
        for date, day in state.day_plans.items():
            if date == target_date:
                continue
            for stop in day.stops:
                place_id = _place_id_from_candidate_id(stop.candidate_id)
                if place_id is not None:
                    visited_place_ids.append(place_id)

        day_plans = dict(state.day_plans)
        day_plans.pop(target_date, None)

        temp_state = state.model_copy(
            update={
                "day_plans": day_plans,
                "visited_place_ids": visited_place_ids,
                "start_date": target_date,
                "num_days": 1,
            }
        )
        events, data, _ = await self.plan_multi_day(
            request, temp_state, model, tools
        )

        # Merge the replanned day back into the original state shape.
        final_day_plans = dict(state.day_plans)
        final_visited = list(state.visited_place_ids)
        if data.items:
            new_day = DayPlan(
                date=target_date,
                anchor_place=state.anchor_place or "Manhattan",
                anchor_time=f"{target_date}T10:00:00",
                theme=f"modified: {state.additional_context or ''}",
                stops=data.items,
            )
            final_day_plans[target_date] = new_day
            for item in data.items:
                place_id = _place_id_from_candidate_id(item.candidate_id)
                if place_id is not None and place_id not in final_visited:
                    final_visited.append(place_id)

        merged_items = _merge_day_stops(final_day_plans)
        full_data = RecommendationData(
            source="itinerary",
            items=merged_items,
            start_date=state.start_date,
            end_date=_trip_end_date(state),
            summary=_summarize_plan(
                state.start_date or target_date,
                state.num_days or 1,
                merged_items,
            ),
        )
        updated_state = state.model_copy(
            update={
                "mode": "multi",
                "day_plans": final_day_plans,
                "visited_place_ids": final_visited,
            }
        )
        return events, full_data, updated_state


def _find_tool(tools: tuple[BaseTool, ...], name: str) -> BaseTool | None:
    for tool in tools:
        if tool.name == name:
            return tool
    return None


async def _call_itinerary_tool(
    tool: BaseTool,
    anchor_place: str,
    anchor_time: str,
    additional_context: str,
    request: Any,
) -> ToolResponse:
    config = _tool_config_from_request(request)
    try:
        result = await tool.ainvoke(
            {
                "anchor_place": anchor_place,
                "anchor_time": anchor_time,
                "duration_hours": 8,
                "additional_context": additional_context,
            },
            config,
        )
    except Exception as exc:
        logger.warning(
            "plan_itinerary_failed",
            extra={"anchor_place": anchor_place, "anchor_time": anchor_time},
            exc_info=exc,
        )
        return ToolResponse(
            status=ToolStatus.ERROR,
            summary="Itinerary planning failed.",
            data={},
        )

    if isinstance(result, ToolResponse):
        return result
    if isinstance(result, str):
        try:
            return ToolResponse.model_validate_json(result)
        except Exception:
            return ToolResponse(
                status=ToolStatus.SUCCESS,
                summary="plan_itinerary returned text.",
                data={"content": result},
            )
    return ToolResponse(
        status=ToolStatus.SUCCESS,
        summary="plan_itinerary returned a result.",
        data={"content": str(result)},
    )


def _theme_for_day(offset: int, base_context: str) -> str:
    if offset == 0:
        return base_context or "first day"
    return f"day {offset + 1}"


def _build_day_context(theme: str, visited_place_ids: list[str], base_context: str) -> str:
    parts: list[str] = []
    if theme:
        parts.append(f"Theme: {theme}")
    if base_context:
        parts.append(base_context)
    if visited_place_ids:
        parts.append(
            "Avoid these already-visited places: " + ", ".join(visited_place_ids)
        )
    return "; ".join(parts)


def _parse_stops(
    data: Mapping[str, Any], visited_place_ids: Sequence[str]
) -> tuple[list[RecommendationItem], list[str]]:
    stops: list[RecommendationItem] = []
    new_place_ids: list[str] = []
    raw_stops = data.get("stops") if isinstance(data, Mapping) else None
    if not isinstance(raw_stops, list):
        return stops, new_place_ids

    for raw in raw_stops:
        if not isinstance(raw, Mapping):
            continue
        place_id = _as_string(raw.get("place_id"))
        if place_id is None:
            continue
        if place_id in visited_place_ids:
            continue
        candidate_id = _as_string(raw.get("candidate_id")) or f"itinerary:{place_id}"
        name = _as_string(raw.get("place_name"))
        lat = _finite_number(raw.get("lat"))
        lng = _finite_number(
            raw.get("lon") if raw.get("lon") is not None else raw.get("lng")
        )
        if name is None or lat is None or lng is None:
            continue
        stops.append(
            RecommendationItem(
                candidate_id=candidate_id,
                source="itinerary",
                name=name,
                lat=lat,
                lng=lng,
                subtitle=_format_subtitle(raw),
                detail=_format_detail(raw),
                rank=len(stops) + 1,
                reason=_as_string(raw.get("why_recommended")) or "",
            )
        )
        new_place_ids.append(place_id)

    return stops, new_place_ids


def _format_subtitle(raw: Mapping[str, Any]) -> str:
    parts = [
        _as_string(raw.get("time")),
        _as_string(raw.get("neighborhood")),
    ]
    return " · ".join(part for part in parts if part)


def _format_detail(raw: Mapping[str, Any]) -> str:
    parts = [
        _as_string(raw.get("category")),
        _as_string(raw.get("crowd_category")),
        _as_string(raw.get("hours")),
    ]
    return " · ".join(part for part in parts if part)


def _merge_day_stops(day_plans: dict[str, DayPlan]) -> list[RecommendationItem]:
    merged: list[RecommendationItem] = []
    for date in sorted(day_plans):
        day = day_plans[date]
        for stop in day.stops:
            merged.append(stop.model_copy(update={"rank": len(merged) + 1}))
    return merged


def _summarize_plan(
    start_date: str | None, num_days: int, items: list[RecommendationItem]
) -> str:
    if not items:
        return "Your trip plan is ready."

    date_label = _format_date_range(start_date, num_days)
    names = ", ".join(item.name for item in items[:5])
    if len(items) > 5:
        names += f" and {len(items) - 5} more"
    return f"Your {date_label} plan includes {names}."


def _format_date_range(start_date: str | None, num_days: int) -> str:
    if start_date is None:
        return f"{num_days}-day"
    try:
        start = datetime.date.fromisoformat(start_date)
    except ValueError:
        return f"{num_days}-day"
    end = start + datetime.timedelta(days=num_days - 1)
    if start.year == end.year:
        return f"{start:%b %d} – {end:%b %d}"
    return f"{start:%b %d, %Y} – {end:%b %d, %Y}"


def _resolve_target_date(target: str, state: TripState) -> str | None:
    target_lower = target.strip().lower()
    match = re.match(r"day\s*(\d+)", target_lower)
    if match and state.start_date:
        day_index = int(match.group(1)) - 1
        if day_index < 0:
            return None
        return _offset_date(state.start_date, day_index)
    try:
        datetime.date.fromisoformat(target)
        return target
    except ValueError:
        pass
    return None


def _trip_end_date(state: TripState) -> str | None:
    if state.start_date is None or state.num_days is None:
        return None
    return _offset_date(state.start_date, state.num_days - 1)


def _place_id_from_candidate_id(candidate_id: str) -> str | None:
    prefix = "itinerary:"
    if candidate_id.startswith(prefix):
        return candidate_id[len(prefix) :]
    return None


def _offset_date(iso_date: str, days: int) -> str:
    base = datetime.date.fromisoformat(iso_date)
    return (base + datetime.timedelta(days=days)).isoformat()


def _as_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _finite_number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if numeric == numeric and numeric not in {float("inf"), float("-inf")}:
            return numeric
    return None
