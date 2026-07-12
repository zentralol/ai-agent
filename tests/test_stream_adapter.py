"""Tests for structured candidate registration and recommendation selection."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import ToolMessage

from app.agent.stream_adapter import LangChainStreamAdapter
from app.schemas.tools import ToolResponse, ToolStatus
from app.tools.attractions import GET_NEAREST_ATTRACTIONS_TOOL_NAME
from app.tools.itinerary import PLAN_ITINERARY_TOOL_NAME
from app.tools.places import GET_NEARBY_PLACES_TOOL_NAME
from app.tools.recommendations import SELECT_RECOMMENDED_PLACES_TOOL_NAME
from app.tools.recommendations_itinerary import RECOMMEND_TOOL_NAME

_ITINERARY_TOOL_INPUT = {
    "anchor_place": "Greenwich Village",
    "anchor_time": "2026-07-06T10:00:00",
    "duration_hours": 6,
}


def _tool_end_event(
    tool_name: str,
    response: ToolResponse,
    tool_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "output": ToolMessage(
            content=response.model_dump_json(),
            name=tool_name,
            tool_call_id="call-1",
        )
    }
    if tool_input is not None:
        data["input"] = tool_input
    return {
        "event": "on_tool_end",
        "name": tool_name,
        "data": data,
    }


def _nearby_response() -> ToolResponse:
    return ToolResponse(
        status=ToolStatus.SUCCESS,
        summary="Found places.",
        data={
            "places": [
                {
                    "candidate_id": "google:place-a",
                    "name": "Place A",
                    "address": "1 Main St",
                    "primary_type": "Cafe",
                    "lat": 40.7,
                    "lng": -73.9,
                    "rating": 4.5,
                    "distance_km": 0.2,
                },
                {
                    "candidate_id": "google:place-c",
                    "name": "Place C",
                    "address": "3 Main St",
                    "primary_type": "Cafe",
                    "lat": 40.71,
                    "lng": -73.91,
                    "rating": 4.2,
                    "distance_km": 0.4,
                },
            ]
        },
    )


def test_recommendations_use_validated_ids_and_selection_order() -> None:
    adapter = LangChainStreamAdapter()
    adapter.to_zentra_events(
        _tool_end_event(GET_NEARBY_PLACES_TOOL_NAME, _nearby_response())
    )

    selection = ToolResponse(
        status=ToolStatus.SUCCESS,
        summary="Selected places.",
        data={
            "recommendations": [
                {"candidate_id": "google:place-c", "reason": "Quieter"},
                {"candidate_id": "missing", "reason": "Invalid"},
                {"candidate_id": "google:place-c", "reason": "Duplicate"},
                {"candidate_id": "google:place-a", "reason": "Closer"},
            ]
        },
    )
    adapter.to_zentra_events(
        _tool_end_event(SELECT_RECOMMENDED_PLACES_TOOL_NAME, selection)
    )

    data = adapter.recommendation_data
    assert data is not None
    assert [item.candidate_id for item in data.items] == [
        "google:place-c",
        "google:place-a",
    ]
    assert [item.rank for item in data.items] == [1, 2]
    assert [item.reason for item in data.items] == ["Quieter", "Closer"]
    assert adapter.recommendation_event() is not None
    assert adapter.ui_parts()[0]["type"] == "data-places"


def test_attraction_candidates_are_registered_as_attractions() -> None:
    adapter = LangChainStreamAdapter()
    response = ToolResponse(
        status=ToolStatus.SUCCESS,
        summary="Found attractions.",
        data={
            "attractions": [
                {
                    "candidate_id": "attraction:1",
                    "name": "The High Line",
                    "neighborhood": "Chelsea",
                    "category": "Park",
                    "lat": 40.748,
                    "lng": -74.0048,
                    "distance_km": 1.2,
                }
            ]
        },
    )
    adapter.to_zentra_events(
        _tool_end_event(GET_NEAREST_ATTRACTIONS_TOOL_NAME, response)
    )
    adapter.to_zentra_events(
        _tool_end_event(
            SELECT_RECOMMENDED_PLACES_TOOL_NAME,
            ToolResponse(
                status=ToolStatus.SUCCESS,
                summary="Selected one.",
                data={
                    "recommendations": [
                        {"candidate_id": "attraction:1", "reason": "Scenic"}
                    ]
                },
            ),
        )
    )

    data = adapter.recommendation_data
    assert data is not None
    assert data.source == "attractions"
    assert data.items[0].subtitle == "Chelsea"


def _recommend_response() -> ToolResponse:
    return ToolResponse(
        status=ToolStatus.SUCCESS,
        summary="2 recommendations returned.",
        data={
            "candidates": [
                {
                    "candidate_id": "recommend:fort-tryon",
                    "name": "Fort Tryon Park",
                    "lat": 40.8617,
                    "lng": -73.9326,
                    "neighborhood": "Washington Heights",
                    "category": "park",
                    "crowd_category": "Very quiet",
                    "hours": "Open until 1:00 AM",
                },
                {
                    "candidate_id": "recommend:joes-pub",
                    "name": "Joe's Pub",
                    "lat": 40.7282,
                    "lng": -73.9925,
                    "neighborhood": "Greenwich Village",
                    "category": "bar",
                    "crowd_category": "Moderate",
                    "hours": "6:00 PM – 2:00 AM",
                },
            ]
        },
    )


def test_recommend_candidates_are_registered_as_recommend() -> None:
    adapter = LangChainStreamAdapter()
    adapter.to_zentra_events(
        _tool_end_event(RECOMMEND_TOOL_NAME, _recommend_response())
    )
    adapter.to_zentra_events(
        _tool_end_event(
            SELECT_RECOMMENDED_PLACES_TOOL_NAME,
            ToolResponse(
                status=ToolStatus.SUCCESS,
                summary="Selected one.",
                data={
                    "recommendations": [
                        {
                            "candidate_id": "recommend:fort-tryon",
                            "reason": "Peaceful escape",
                        }
                    ]
                },
            ),
        )
    )

    data = adapter.recommendation_data
    assert data is not None
    assert data.source == "recommend"
    assert data.items[0].name == "Fort Tryon Park"
    assert data.items[0].subtitle == "Washington Heights"
    assert data.items[0].detail == "park · Very quiet · Open until 1:00 AM"


def test_infer_recommendations_from_text_backfills_backend_cards() -> None:
    adapter = LangChainStreamAdapter()
    adapter.to_zentra_events(
        _tool_end_event(RECOMMEND_TOOL_NAME, _recommend_response())
    )

    adapter.infer_recommendations_from_text(
        "Try Fort Tryon Park for a quiet evening, or Joe's Pub for live music."
    )

    data = adapter.recommendation_data
    assert data is not None
    assert data.source == "recommend"
    assert [item.name for item in data.items] == ["Fort Tryon Park", "Joe's Pub"]
    assert [item.rank for item in data.items] == [1, 2]


def _itinerary_response() -> ToolResponse:
    return ToolResponse(
        status=ToolStatus.SUCCESS,
        summary="Itinerary built: 2 stops.",
        data={
            "stops": [
                {
                    "time": "16:00",
                    "place_id": "washington-square",
                    "place_name": "Washington Square Park",
                    "candidate_id": "itinerary:washington-square",
                    "lat": 40.7308,
                    "lon": -73.9973,
                    "neighborhood": "Greenwich Village",
                    "category": "park",
                    "crowd_category": "Very busy",
                    "hours": "Open 24 hours",
                    "why_recommended": "Historic park stroll",
                },
                {
                    "time": "20:10",
                    "place_id": "essex-market",
                    "place_name": "Essex Market",
                    "candidate_id": "itinerary:essex-market",
                    "lat": 40.7185,
                    "lon": -73.9877,
                    "neighborhood": "Lower East Side",
                    "category": "food",
                    "crowd_category": "Moderate",
                    "hours": "08:00-21:00",
                    "why_recommended": "Vegetarian-friendly dinner",
                },
            ],
            "candidates": [
                {
                    "candidate_id": "itinerary:washington-square",
                    "name": "Washington Square Park",
                    "lat": 40.7308,
                    "lng": -73.9973,
                    "time": "16:00",
                    "neighborhood": "Greenwich Village",
                    "category": "park",
                    "crowd_category": "Very busy",
                    "hours": "Open 24 hours",
                    "why_recommended": "Historic park stroll",
                },
                {
                    "candidate_id": "itinerary:essex-market",
                    "name": "Essex Market",
                    "lat": 40.7185,
                    "lng": -73.9877,
                    "time": "20:10",
                    "neighborhood": "Lower East Side",
                    "category": "food",
                    "crowd_category": "Moderate",
                    "hours": "08:00-21:00",
                    "why_recommended": "Vegetarian-friendly dinner",
                },
            ],
        },
    )


def test_itinerary_auto_selects_all_stops_in_order() -> None:
    adapter = LangChainStreamAdapter()
    adapter.to_zentra_events(
        _tool_end_event(
            PLAN_ITINERARY_TOOL_NAME,
            _itinerary_response(),
            _ITINERARY_TOOL_INPUT,
        )
    )

    data = adapter.recommendation_data
    assert data is not None
    assert data.source == "itinerary"
    assert [item.name for item in data.items] == [
        "Washington Square Park",
        "Essex Market",
    ]
    assert data.target_time == "2026-07-06T10:00:00"
    assert [item.rank for item in data.items] == [1, 2]
    assert data.items[0].subtitle == "Jul 6, 2026, 4:00 PM · Greenwich Village"
    assert data.items[1].subtitle == "Jul 6, 2026, 8:10 PM · Lower East Side"
    assert data.target_time == "2026-07-06T10:00:00"
    assert data.items[0].detail == "park · Very busy · Open 24 hours"
    assert data.items[0].reason == "Historic park stroll"


def test_infer_recommendations_from_text_backfills_itinerary_cards() -> None:
    adapter = LangChainStreamAdapter()
    response = _itinerary_response()
    response.data.pop("stops", None)
    adapter.to_zentra_events(
        _tool_end_event(PLAN_ITINERARY_TOOL_NAME, response, _ITINERARY_TOOL_INPUT)
    )

    adapter.infer_recommendations_from_text(
        "Start at Washington Square Park, then dinner at Essex Market."
    )

    data = adapter.recommendation_data
    assert data is not None
    assert data.source == "itinerary"
    assert [item.name for item in data.items] == [
        "Washington Square Park",
        "Essex Market",
    ]

def test_itinerary_target_time_falls_back_to_result_anchor_time() -> None:
    adapter = LangChainStreamAdapter()
    response = _itinerary_response()
    response.data["anchor_time"] = "2026-07-06T10:00:00"
    adapter.to_zentra_events(_tool_end_event(PLAN_ITINERARY_TOOL_NAME, response))

    data = adapter.recommendation_data
    assert data is not None
    assert data.target_time == "2026-07-06T10:00:00"


def test_itinerary_select_preserves_target_time() -> None:
    adapter = LangChainStreamAdapter()
    response = _itinerary_response()
    response.data["anchor_time"] = "2026-07-06T10:00:00"
    adapter.to_zentra_events(
        _tool_end_event(
            PLAN_ITINERARY_TOOL_NAME,
            response,
            _ITINERARY_TOOL_INPUT,
        )
    )
    adapter.to_zentra_events(
        _tool_end_event(
            SELECT_RECOMMENDED_PLACES_TOOL_NAME,
            ToolResponse(
                status=ToolStatus.SUCCESS,
                summary="Selected stops.",
                data={
                    "recommendations": [
                        {
                            "candidate_id": "itinerary:washington-square",
                            "reason": "Start here",
                        },
                        {
                            "candidate_id": "itinerary:essex-market",
                            "reason": "Dinner",
                        },
                    ]
                },
            ),
        )
    )

    data = adapter.recommendation_data
    assert data is not None
    assert data.source == "itinerary"
    assert data.target_time == "2026-07-06T10:00:00"
    assert [item.name for item in data.items] == [
        "Washington Square Park",
        "Essex Market",
    ]
