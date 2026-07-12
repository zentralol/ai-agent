"""Tests for recommendation stream schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.recommendations import RecommendationData, RecommendationItem


def _item() -> RecommendationItem:
    return RecommendationItem(
        candidate_id="itinerary:essex-market",
        source="itinerary",
        name="Essex Market",
        lat=40.7185,
        lng=-73.9877,
        rank=1,
    )


def test_recommendation_data_accepts_valid_target_time() -> None:
    data = RecommendationData(
        source="itinerary",
        items=[_item()],
        target_time="2026-07-10T16:00:00",
    )
    assert data.target_time == "2026-07-10T16:00:00"


def test_recommendation_data_normalizes_target_time_without_seconds() -> None:
    data = RecommendationData(
        source="itinerary",
        items=[_item()],
        target_time="2026-07-10T16:00",
    )
    assert data.target_time == "2026-07-10T16:00:00"


def test_recommendation_data_rejects_time_only_target_time() -> None:
    with pytest.raises(ValidationError):
        RecommendationData(
            source="itinerary",
            items=[_item()],
            target_time="16:00:00",
        )


def test_recommendation_data_rejects_date_only_target_time() -> None:
    with pytest.raises(ValidationError):
        RecommendationData(
            source="itinerary",
            items=[_item()],
            target_time="2026-07-10",
        )


def test_recommendation_data_allows_null_target_time() -> None:
    data = RecommendationData(source="nearby", items=[_item()])
    assert data.target_time is None
