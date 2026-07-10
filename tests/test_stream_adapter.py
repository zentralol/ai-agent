"""Tests for structured candidate registration and recommendation selection."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import ToolMessage

from app.agent.stream_adapter import LangChainStreamAdapter
from app.schemas.tools import ToolResponse, ToolStatus
from app.tools.attractions import GET_NEAREST_ATTRACTIONS_TOOL_NAME
from app.tools.places import GET_NEARBY_PLACES_TOOL_NAME
from app.tools.recommendations import SELECT_RECOMMENDED_PLACES_TOOL_NAME


def _tool_end_event(tool_name: str, response: ToolResponse) -> dict[str, Any]:
    return {
        "event": "on_tool_end",
        "name": tool_name,
        "data": {
            "output": ToolMessage(
                content=response.model_dump_json(),
                name=tool_name,
                tool_call_id="call-1",
            )
        },
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
