"""Tests for TripState serialization and helpers."""

from __future__ import annotations

from app.agent.trip_state import (
    ClarificationState,
    DayPlan,
    TripState,
    from_dict,
    merge_answer,
    to_dict,
)
from app.schemas.recommendations import RecommendationItem


def test_default_trip_state_round_trip() -> None:
    state = TripState()

    data = to_dict(state)
    restored = from_dict(data)

    assert restored == state
    assert restored.version == 1
    assert restored.mode is None
    assert restored.num_days is None
    assert restored.start_date is None
    assert restored.anchor_place is None
    assert restored.additional_context == ""
    assert restored.day_plans == {}
    assert restored.visited_place_ids == []
    assert restored.clarification is None


def test_trip_state_with_day_plans_round_trip() -> None:
    stop = RecommendationItem(
        candidate_id="itinerary:central-park",
        source="itinerary",
        name="Central Park",
        lat=40.785091,
        lng=-73.968285,
        rank=1,
        reason="Morning walk",
    )
    day = DayPlan(
        date="2026-07-12",
        anchor_place="Central Park",
        anchor_time="2026-07-12T10:00:00",
        theme="parks",
        stops=[stop],
    )
    state = TripState(
        mode="multi",
        num_days=1,
        start_date="2026-07-12",
        anchor_place="Central Park",
        additional_context="relaxed pace",
        day_plans={"2026-07-12": day},
        visited_place_ids=["central-park"],
        clarification=ClarificationState(missing=["budget"], count=1),
    )

    data = to_dict(state)
    restored = from_dict(data)

    assert restored == state
    assert restored.day_plans["2026-07-12"].stops[0].name == "Central Park"


def test_from_dict_returns_default_for_none() -> None:
    state = from_dict(None)

    assert state == TripState()


def test_from_dict_ignores_unknown_fields() -> None:
    data = {"version": 1, "mode": "multi", "unknown_field": "ignore"}
    state = from_dict(data)

    assert state.mode == "multi"
    assert not hasattr(state, "unknown_field")


def test_merge_answer_folds_clarification_response() -> None:
    state = TripState(
        clarification=ClarificationState(missing=["num_days"], count=1),
    )

    updated = merge_answer(state, "3 days")

    assert updated.num_days == 3
    assert updated.clarification is None


def test_merge_answer_increments_count_when_still_missing() -> None:
    state = TripState(
        clarification=ClarificationState(missing=["start_date"], count=1),
    )

    updated = merge_answer(state, "I don't know")

    assert updated.clarification is not None
    assert updated.clarification.count == 2
    assert "start_date" in updated.clarification.missing


def test_merge_answer_caps_clarification_at_two_rounds() -> None:
    state = TripState(
        num_days=3,
        clarification=ClarificationState(missing=["start_date"], count=2),
    )

    updated = merge_answer(state, "still don't know")

    assert updated.clarification is None
