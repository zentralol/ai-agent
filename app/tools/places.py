"""Nearby places tool backed by Google Maps Platform Places API (New).

Finds businesses and points of interest (cafes, restaurants, shops, malls, bars,
etc.) around the user's current device location. The model supplies *what* to
look for as a free-text ``query``; *where* comes from the device coordinates in
the runtime config and is never returned to the model.
"""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from time import perf_counter
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.config import Settings, get_settings
from app.geo import as_float, configurable_float, haversine_km
from app.schemas.tools import ToolResponse, ToolStatus

GET_NEARBY_PLACES_TOOL_NAME = "get_nearby_places"

PLACES_SEARCH_TEXT_URL = "https://places.googleapis.com/v1/places:searchText"
PLACES_FIELD_MASK = ",".join(
    (
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.location",
        "places.primaryTypeDisplayName",
        "places.rating",
        "places.userRatingCount",
        "places.currentOpeningHours.openNow",
        "places.priceLevel",
    )
)
HTTP_TIMEOUT_SECONDS = 8.0

# Number of places returned to the model per search.
MAX_RESULTS = 8

# A Places circle bias requires a radius; we use the API maximum (50 km) so the
# search is centered on the user without imposing a range limit. Text Search
# still ranks the nearest, most relevant results first.
LOCATION_BIAS_RADIUS_M = 50000.0

logger = logging.getLogger("zentra_agent.tools.places")


@tool
async def get_nearby_places(query: str, config: RunnableConfig) -> str:
    """Find businesses and points of interest near the user's current location.

    Use this for things like cafes, restaurants, bars, shops, or malls. Pass what
    to look for as ``query`` (for example "coffee", "ramen", "shopping mall").
    The user's location is supplied by their device, not by you.
    """

    request_id = _configurable_string(config, "request_id")
    conversation_id = _configurable_string(config, "conversation_id")
    lat = configurable_float(config, "lat")
    lng = configurable_float(config, "lng")
    cleaned_query = query.strip()
    started_at = perf_counter()

    logger.info(
        "tool_call_start tool=%s request_id=%s conversation_id=%s has_location=%s has_query=%s",
        GET_NEARBY_PLACES_TOOL_NAME,
        request_id,
        conversation_id,
        lat is not None and lng is not None,
        bool(cleaned_query),
    )

    if not cleaned_query:
        result = ToolResponse(
            status=ToolStatus.WARNING,
            summary="No search term was provided.",
            next_actions=["Ask the user what kind of place they are looking for."],
        )
    elif lat is None or lng is None:
        result = ToolResponse(
            status=ToolStatus.WARNING,
            summary="The user's current location is not available.",
            next_actions=[
                "Ask the user to share their location or name a place to search near."
            ],
        )
    else:
        result = await get_places_tool().search_nearby(cleaned_query, lat, lng)

    logger.info(
        "tool_call_end tool=%s status=%s request_id=%s conversation_id=%s duration_ms=%.2f",
        GET_NEARBY_PLACES_TOOL_NAME,
        result.status.value,
        request_id,
        conversation_id,
        _duration_ms(started_at),
    )
    return result.model_dump_json()


class PlacesTool:
    """Query the Places API and shape results for the model."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    async def search_nearby(self, query: str, lat: float, lng: float) -> ToolResponse:
        """Return places matching ``query`` near ``(lat, lng)``."""

        api_key = self._settings.google_maps_api_key
        if not api_key:
            logger.warning("places_tool_unconfigured missing=GOOGLE_MAPS_API_KEY")
            return ToolResponse(
                status=ToolStatus.WARNING,
                summary="Nearby place search is unavailable; the maps API is not configured.",
                next_actions=["Continue without place data or ask a clarifying question."],
            )

        try:
            places = await self._request(api_key, query, lat, lng)
        except httpx.HTTPError as exc:
            logger.exception(
                "places_tool_request_failed error_type=%s", type(exc).__name__
            )
            return ToolResponse(
                status=ToolStatus.ERROR,
                summary="Failed to search for nearby places.",
                next_actions=["Continue without place data for this response."],
            )

        ranked = _shape_places(places, lat, lng)
        return ToolResponse(
            status=ToolStatus.SUCCESS,
            summary=f"Found {len(ranked)} places for '{query}'.",
            data={"places": ranked, "query": query},
        )

    async def _request(
        self, api_key: str, query: str, lat: float, lng: float
    ) -> list[dict[str, Any]]:
        client = self._get_client()
        started_at = perf_counter()
        response = await client.post(
            PLACES_SEARCH_TEXT_URL,
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": api_key,
                "X-Goog-FieldMask": PLACES_FIELD_MASK,
            },
            json={
                "textQuery": query,
                "locationBias": {
                    "circle": {
                        "center": {"latitude": lat, "longitude": lng},
                        "radius": LOCATION_BIAS_RADIUS_M,
                    }
                },
                "maxResultCount": MAX_RESULTS,
                "languageCode": "en",
            },
        )
        response.raise_for_status()
        payload = response.json()
        places = payload.get("places") if isinstance(payload, dict) else None
        logger.info(
            "places_tool_request_end count=%d duration_ms=%.2f",
            len(places) if isinstance(places, list) else 0,
            _duration_ms(started_at),
        )
        return [p for p in places if isinstance(p, dict)] if isinstance(places, list) else []

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS)
        return self._client


@lru_cache(maxsize=1)
def get_places_tool() -> PlacesTool:
    """Return the cached places lookup tool."""

    return PlacesTool(get_settings())


def _shape_places(
    places: list[dict[str, Any]], lat: float, lng: float
) -> list[dict[str, Any]]:
    shaped: list[dict[str, Any]] = []
    for place in places:
        # The place's own coordinates are public and returned so the client can
        # offer navigation. The user's coordinates are only used here to compute
        # a relative distance and are never returned.
        location = place.get("location")
        location = location if isinstance(location, dict) else {}
        place_lat = as_float(location.get("latitude"))
        place_lng = as_float(location.get("longitude"))
        distance_km = (
            round(haversine_km(lat, lng, place_lat, place_lng), 2)
            if place_lat is not None and place_lng is not None
            else None
        )
        shaped.append(
            {
                "candidate_id": _candidate_id(
                    place.get("id"),
                    _nested_text(place.get("displayName")),
                    place_lat,
                    place_lng,
                ),
                "name": _nested_text(place.get("displayName")),
                "address": _as_string(place.get("formattedAddress")),
                "primary_type": _nested_text(place.get("primaryTypeDisplayName")),
                "lat": place_lat,
                "lng": place_lng,
                "rating": place.get("rating"),
                "rating_count": place.get("userRatingCount"),
                "open_now": _open_now(place.get("currentOpeningHours")),
                "price_level": _as_string(place.get("priceLevel")) or None,
                "distance_km": distance_km,
            }
        )
    return shaped


def _candidate_id(
    external_id: object,
    name: str,
    lat: float | None,
    lng: float | None,
) -> str:
    if isinstance(external_id, str) and external_id.strip():
        return f"google:{external_id.removeprefix('places/').strip()}"
    fallback = f"{name.strip().lower()}|{lat}|{lng}"
    digest = hashlib.sha256(fallback.encode("utf-8")).hexdigest()[:20]
    return f"google:fallback-{digest}"


def _nested_text(value: object) -> str:
    if isinstance(value, dict):
        return _as_string(value.get("text"))
    return ""


def _open_now(opening_hours: object) -> bool | None:
    if not isinstance(opening_hours, dict):
        return None
    value = opening_hours.get("openNow")
    return value if isinstance(value, bool) else None


def _configurable_string(config: RunnableConfig, key: str) -> str | None:
    value = config.get("configurable", {}).get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _as_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _duration_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000
