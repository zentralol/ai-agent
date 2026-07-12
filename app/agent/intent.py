"""Classify the user's intent for trip planning requests."""

from __future__ import annotations

import datetime
import re
from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agent.trip_state import TripState, merge_answer

Intent = Literal[
    "single_day",
    "multi_day",
    "modify_day",
    "clarify",
    "out_of_scope",
    "general",
]


class IntentClassification(BaseModel):
    """Structured output for intent classification."""

    intent: Intent
    num_days: int | None = Field(default=None, ge=1, le=14)
    start_date: str | None = None
    anchor_place: str | None = None
    modify_target: str | None = None
    additional_context: str = ""
    missing_fields: list[str] = []
    question_to_user: str = ""


_INTENT_SYSTEM_PROMPT = (
    "You are an intent classifier for a travel assistant. Given the user's message, "
    "classify the intent into one of: single_day, multi_day, modify_day, clarify, "
    "out_of_scope, general. Extract any trip parameters (number of days, start date, "
    "anchor place, day to modify) when present. If required information is missing, "
    "set intent to clarify and list the missing fields. Always respond in English."
)


async def classify_intent(
    model: BaseChatModel,
    message: str,
    current_state: TripState,
    today: str,
) -> IntentClassification:
    """Classify user intent and extract planning parameters.

    Applies deterministic guardrails on top of the model's structured output
    so common date/day patterns are parsed reliably. Caps clarification at
    two rounds; after that the agent proceeds with sensible defaults.
    """

    guardrails = _extract_guardrails(message, today)

    if current_state.clarification is not None:
        updated_state = merge_answer(current_state, message, today)
        if updated_state.clarification is None:
            return _intent_from_state(updated_state, "multi_day", today)
        return IntentClassification(
            intent="clarify",
            missing_fields=updated_state.clarification.missing,
            question_to_user=_default_clarification_question(
                updated_state.clarification.missing
            ),
        )

    if not _looks_like_planning_request(message, today):
        return IntentClassification(intent="general")

    structured_model = model.with_structured_output(IntentClassification)
    try:
        response = await structured_model.ainvoke(
            [
                SystemMessage(content=_INTENT_SYSTEM_PROMPT),
                HumanMessage(content=message),
            ]
        )
    except OutputParserException:
        response = None
    if not isinstance(response, IntentClassification):
        response = IntentClassification(intent="general")

    result = _apply_guardrails(response, guardrails)

    if result.intent in {"clarify", "multi_day"} and (
        result.missing_fields or result.intent == "clarify"
    ):
        clarification_count = (
            current_state.clarification.count or 0
            if current_state.clarification
            else 0
        )
        if clarification_count >= 2:
            result = _force_multi_day_defaults(result, today)

    return result


def _extract_guardrails(message: str, today: str) -> dict[str, object]:
    guardrails: dict[str, object] = {}

    num_days = _extract_num_days(message)
    if num_days is not None:
        guardrails["num_days"] = num_days

    start_date = _extract_start_date(message, today)
    if start_date is not None:
        guardrails["start_date"] = start_date

    modify_target = _extract_modify_target(message)
    if modify_target is not None:
        guardrails["modify_target"] = modify_target

    anchor_place = _extract_anchor_place(message)
    if anchor_place is not None:
        guardrails["anchor_place"] = anchor_place

    return guardrails


def _apply_guardrails(
    result: IntentClassification, guardrails: dict[str, object]
) -> IntentClassification:
    updates: dict[str, object] = {}
    if "num_days" in guardrails and result.num_days is None:
        updates["num_days"] = guardrails["num_days"]
    if "start_date" in guardrails and result.start_date is None:
        updates["start_date"] = guardrails["start_date"]
    if "modify_target" in guardrails and result.modify_target is None:
        updates["modify_target"] = guardrails["modify_target"]
    if "anchor_place" in guardrails and result.anchor_place is None:
        updates["anchor_place"] = guardrails["anchor_place"]

    if result.intent in {"clarify", "multi_day"} and result.missing_fields:
        still_missing = [
            field
            for field in result.missing_fields
            if (
                (field != "num_days" or "num_days" not in guardrails)
                and (field != "start_date" or "start_date" not in guardrails)
                and (field != "anchor_place" or "anchor_place" not in guardrails)
            )
        ]
        if not still_missing:
            updates["intent"] = "multi_day"
            updates["missing_fields"] = []
        else:
            updates["missing_fields"] = still_missing

    if updates:
        return result.model_copy(update=updates)
    return result


def _force_multi_day_defaults(
    result: IntentClassification, today: str
) -> IntentClassification:
    updates: dict[str, object] = {"intent": "multi_day", "missing_fields": []}
    if result.num_days is None:
        updates["num_days"] = 3
    if result.start_date is None:
        updates["start_date"] = _offset_date(today, 1)
    if result.anchor_place is None:
        updates["anchor_place"] = result.anchor_place or "Manhattan"
    return result.model_copy(update=updates)


def _extract_num_days(text: str) -> int | None:
    match = re.search(r"\b(\d+)\s*(?:day|days)\b", text, re.IGNORECASE)
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

    month_day = re.search(
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})\b",
        lowered,
    )
    if month_day:
        month_name = month_day.group(1).capitalize()
        day = int(month_day.group(2))
        today_date = datetime.date.fromisoformat(today)
        year = today_date.year
        try:
            parsed = datetime.datetime.strptime(
                f"{year}-{month_name}-{day}", "%Y-%b-%d"
            ).date()
            if parsed < today_date:
                parsed = datetime.datetime.strptime(
                    f"{year + 1}-{month_name}-{day}", "%Y-%b-%d"
                ).date()
            return parsed.isoformat()
        except ValueError:
            pass

    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if iso_match:
        candidate = iso_match.group(1)
        try:
            datetime.date.fromisoformat(candidate)
            return candidate
        except ValueError:
            pass

    return None


def _extract_modify_target(text: str) -> str | None:
    match = re.search(
        r"\b(day\s*\d+|\d{4}-\d{2}-\d{2})\b", text, re.IGNORECASE
    )
    if match:
        return match.group(1).strip().lower()
    return None


def _extract_anchor_place(text: str) -> str | None:
    match = re.search(
        r"(?:in|around|near)\s+([a-zA-Z\s]+?)(?:\s+(?:starting|for|on|from|today|tomorrow|now|$))",
        text,
        re.IGNORECASE,
    )
    if match:
        place = match.group(1).strip()
        return place if place else None
    return None


def _offset_date(iso_date: str, days: int) -> str:
    base = datetime.date.fromisoformat(iso_date)
    return (base + datetime.timedelta(days=days)).isoformat()


def _looks_like_planning_request(message: str, today: str) -> bool:
    lowered = message.lower()
    planning_keywords = (
        "plan",
        "itinerary",
        "trip",
        "days",
        "day",
        "visit",
        "modify",
        "change",
        "replace",
        "add",
        "remove",
        "manhattan",
        "new york",
    )
    if any(keyword in lowered for keyword in planning_keywords):
        return True
    guardrails = _extract_guardrails(message, today)
    return bool(guardrails)


def _intent_from_state(
    state: TripState, default_intent: Intent, today: str
) -> IntentClassification:
    num_days = state.num_days if state.num_days is not None else 3
    start_date = (
        state.start_date if state.start_date is not None else _offset_date(today, 1)
    )
    anchor_place = (
        state.anchor_place if state.anchor_place is not None else "Manhattan"
    )
    return IntentClassification(
        intent=default_intent,
        num_days=num_days,
        start_date=start_date,
        anchor_place=anchor_place,
        additional_context=state.additional_context,
    )


def _default_clarification_question(missing: list[str]) -> str:
    if "num_days" in missing:
        return "How many days would you like to plan?"
    if "start_date" in missing:
        return "What date would you like to start?"
    if "anchor_place" in missing:
        return "Where would you like to start your trip?"
    return "Could you share a bit more detail so I can plan this for you?"
