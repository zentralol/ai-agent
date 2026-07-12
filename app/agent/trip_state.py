"""Pydantic models for the agent's multi-day trip planning state."""

from __future__ import annotations

import datetime
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from app.schemas.recommendations import RecommendationItem


class DayPlan(BaseModel):
    """One day of a multi-day trip plan."""

    model_config = ConfigDict(extra="ignore")

    date: str
    anchor_place: str
    anchor_time: str
    theme: str = ""
    stops: list[RecommendationItem] = []


class ClarificationState(BaseModel):
    """Tracks what the agent still needs to know before planning."""

    model_config = ConfigDict(extra="ignore")

    missing: list[str] = []
    count: int = 0


class TripState(BaseModel):
    """Session-level state for multi-day trip planning."""

    model_config = ConfigDict(extra="ignore")

    version: Literal[1] = 1
    mode: Literal["single", "multi"] | None = None
    num_days: int | None = None
    start_date: str | None = None
    anchor_place: str | None = None
    additional_context: str = ""
    day_plans: dict[str, DayPlan] = {}
    visited_place_ids: list[str] = []
    clarification: ClarificationState | None = None


def to_dict(state: TripState) -> dict[str, Any]:
    """Serialize a TripState to a plain dictionary."""
    return state.model_dump(mode="json")


def from_dict(data: dict[str, Any] | None) -> TripState:
    """Deserialize a TripState from a plain dictionary."""
    if data is None:
        return TripState()
    return TripState.model_validate(data)


def merge_answer(state: TripState, answer: str, today: str | None = None) -> TripState:
    """Fold a clarifying answer from the user back into the state.

    Parses the answer for days, dates, and places. If all currently missing
    fields are resolved, the clarification is cleared. Otherwise the
    clarification count is incremented. After two clarification rounds the
    agent stops asking and falls back to sensible defaults.
    """

    text = answer.strip().lower()
    today = today or datetime.date.today().isoformat()

    updates: dict[str, object] = {}
    if state.num_days is None:
        num_days = _extract_num_days(text)
        if num_days is not None:
            updates["num_days"] = num_days

    if state.start_date is None:
        start_date = _extract_start_date(text, today)
        if start_date is not None:
            updates["start_date"] = start_date

    if state.anchor_place is None:
        place = _extract_anchor_place(text)
        if place is not None:
            updates["anchor_place"] = place

    if updates:
        state = state.model_copy(update=updates)

    clarification = state.clarification
    if clarification is None or not clarification.missing:
        return state.model_copy(update={"clarification": None})

    resolved = _resolved_missing(clarification.missing, state)
    still_missing = [field for field in clarification.missing if field not in resolved]

    if not still_missing:
        return state.model_copy(update={"clarification": None})

    if clarification.count >= 2:
        return _apply_clarification_defaults(
            state, still_missing, today
        ).model_copy(update={"clarification": None})

    return state.model_copy(
        update={
            "clarification": clarification.model_copy(
                update={"missing": still_missing, "count": clarification.count + 1}
            )
        }
    )


def _extract_num_days(text: str) -> int | None:
    match = re.search(r"\b(\d+)\s*(?:day|days)\b", text)
    if match:
        value = int(match.group(1))
        return max(1, min(value, 14))
    return None


def _extract_start_date(text: str, today: str) -> str | None:
    lowered = text.lower()
    if "today" in lowered:
        return today
    if "tomorrow" in lowered:
        return _offset_date(today, 1)

    # Month day, e.g. "July 12" or "Jul 12"
    month_day = re.search(
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})\b",
        lowered,
    )
    if month_day:
        month_name = month_day.group(1).capitalize()
        day = int(month_day.group(2))
        year = datetime.date.fromisoformat(today).year
        try:
            parsed = datetime.datetime.strptime(
                f"{year}-{month_name}-{day}", "%Y-%b-%d"
            ).date()
            return parsed.isoformat()
        except ValueError:
            pass

    # ISO date
    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if iso_match:
        candidate = iso_match.group(1)
        try:
            datetime.date.fromisoformat(candidate)
            return candidate
        except ValueError:
            pass

    return None


def _extract_anchor_place(text: str) -> str | None:
    # Very lightweight extraction: "in X" or "starting in X".
    match = re.search(
        r"(?:in|around|near)\s+([a-z\s]+?)(?:\s+(?:starting|for|on|from|today|tomorrow|now|$))",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().title() or None
    return None


def _resolved_missing(missing: list[str], state: TripState) -> set[str]:
    resolved: set[str] = set()
    for field in missing:
        if field == "num_days" and state.num_days is not None:
            resolved.add(field)
        elif field == "start_date" and state.start_date is not None:
            resolved.add(field)
        elif field == "anchor_place" and state.anchor_place is not None:
            resolved.add(field)
    return resolved


def _apply_clarification_defaults(
    state: TripState, still_missing: list[str], today: str
) -> TripState:
    defaults: dict[str, object] = {}
    if "num_days" in still_missing:
        defaults["num_days"] = 3
    if "start_date" in still_missing:
        defaults["start_date"] = _offset_date(today, 1)
    if "anchor_place" in still_missing and state.anchor_place is None:
        defaults["anchor_place"] = "Manhattan"
    if defaults:
        return state.model_copy(update=defaults)
    return state


def _offset_date(iso_date: str, days: int) -> str:
    base = datetime.date.fromisoformat(iso_date)
    return (base + datetime.timedelta(days=days)).isoformat()
