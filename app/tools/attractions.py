"""Nearest-attractions lookup tool grounded in the user's device location."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from functools import lru_cache
from time import perf_counter
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from postgrest.exceptions import APIError
from supabase import AsyncClient

from app.config import Settings, get_settings
from app.db.supabase_client import create_supabase_client
from app.geo import as_float, configurable_float, haversine_km
from app.schemas.tools import ToolResponse, ToolStatus

GET_NEAREST_ATTRACTIONS_TOOL_NAME = "get_nearest_attractions"

# Fixed to match the product's Supabase schema.
ATTRACTIONS_TABLE = "attractions"
DEFAULT_LIMIT = 5
DESCRIPTION_MAX_CHARS = 160

logger = logging.getLogger("zentra_agent.tools.attractions")


@tool
async def get_nearest_attractions(config: RunnableConfig) -> str:
    """Find tourist attractions closest to the user's current device location.

    Call this with no arguments when the user asks what is nearby or wants
    attraction suggestions near them. Location comes from the shared device
    position, not from you.
    """

    request_id = _configurable_string(config, "request_id")
    conversation_id = _configurable_string(config, "conversation_id")
    lat = configurable_float(config, "lat")
    lng = configurable_float(config, "lng")
    started_at = perf_counter()

    logger.info(
        "tool_call_start tool=%s request_id=%s conversation_id=%s has_location=%s",
        GET_NEAREST_ATTRACTIONS_TOOL_NAME,
        request_id,
        conversation_id,
        lat is not None and lng is not None,
    )

    if lat is None or lng is None:
        result = ToolResponse(
            status=ToolStatus.WARNING,
            summary="The user's current location is not available.",
            next_actions=[
                "Ask the user to share their location or name a place to search near."
            ],
        )
    else:
        result = await get_attractions_tool().nearest(lat=lat, lng=lng)

    logger.info(
        "tool_call_end tool=%s status=%s request_id=%s conversation_id=%s duration_ms=%.2f",
        GET_NEAREST_ATTRACTIONS_TOOL_NAME,
        result.status.value,
        request_id,
        conversation_id,
        _duration_ms(started_at),
    )
    return result.model_dump_json()


class AttractionsTool:
    """Fetch attractions from Supabase and rank them by distance."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncClient | None = None

    async def nearest(
        self, lat: float, lng: float, limit: int = DEFAULT_LIMIT
    ) -> ToolResponse:
        """Return the ``limit`` attractions closest to ``(lat, lng)``."""

        client = await self._get_client()
        if client is None:
            logger.warning("attractions_tool_supabase_unconfigured table=%s", ATTRACTIONS_TABLE)
            return ToolResponse(
                status=ToolStatus.WARNING,
                summary="Nearby attractions are unavailable; the database is not configured.",
                next_actions=["Continue without attraction data or ask a clarifying question."],
            )

        try:
            rows = await self._fetch_rows(client)
        except (APIError, httpx.HTTPError) as exc:
            logger.exception(
                "attractions_tool_query_failed table=%s error_type=%s",
                ATTRACTIONS_TABLE,
                type(exc).__name__,
            )
            return ToolResponse(
                status=ToolStatus.ERROR,
                summary="Failed to load attractions from the database.",
                next_actions=["Continue without attraction data for this response."],
            )

        # The user's raw coordinates are deliberately NOT included in the
        # response: only the ranked attractions (with relative distance) go back
        # to the model, so the LLM never receives the exact device location.
        ranked = _rank_by_distance(rows, lat, lng, limit)
        if not ranked:
            return ToolResponse(
                status=ToolStatus.SUCCESS,
                summary="No attractions with usable coordinates were found.",
                data={"attractions": []},
            )

        return ToolResponse(
            status=ToolStatus.SUCCESS,
            summary=f"Found the {len(ranked)} nearest attractions.",
            data={"attractions": ranked},
        )

    async def _fetch_rows(self, client: AsyncClient) -> list[Mapping[str, Any]]:
        response = await (
            client.table(ATTRACTIONS_TABLE)
            .select("id, Name, Category, Neighborhood, Description, lat, lon")
            .execute()
        )
        rows = response.data if isinstance(response.data, list) else []
        return [row for row in rows if isinstance(row, Mapping)]

    async def _get_client(self) -> AsyncClient | None:
        if self._client is None:
            self._client = await create_supabase_client(self._settings)
        return self._client


@lru_cache(maxsize=1)
def get_attractions_tool() -> AttractionsTool:
    """Return the cached attractions lookup tool."""

    return AttractionsTool(get_settings())


def _rank_by_distance(
    rows: list[Mapping[str, Any]], lat: float, lng: float, limit: int
) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        row_lat = as_float(row.get("lat"))
        row_lng = as_float(row.get("lon"))
        if row_lat is None or row_lng is None:
            continue
        distance = haversine_km(lat, lng, row_lat, row_lng)
        scored.append((distance, _to_attraction(row, distance, row_lat, row_lng)))

    scored.sort(key=lambda item: item[0])
    return [attraction for _, attraction in scored[:limit]]


def _to_attraction(
    row: Mapping[str, Any], distance_km: float, lat: float, lng: float
) -> dict[str, Any]:
    # The attraction's own coordinates are public and returned so the client can
    # offer navigation. The user's coordinates are never returned.
    name = _as_string(row.get("Name"))
    return {
        "candidate_id": _candidate_id(row.get("id"), name, lat, lng),
        "name": name,
        "category": _as_string(row.get("Category")),
        "neighborhood": _as_string(row.get("Neighborhood")),
        "description": _truncate(_as_string(row.get("Description"))),
        "lat": lat,
        "lng": lng,
        "distance_km": round(distance_km, 2),
    }


def _candidate_id(external_id: object, name: str, lat: float, lng: float) -> str:
    if external_id is not None and str(external_id).strip():
        return f"attraction:{str(external_id).strip()}"
    fallback = f"{name.strip().lower()}|{lat}|{lng}"
    digest = hashlib.sha256(fallback.encode("utf-8")).hexdigest()[:20]
    return f"attraction:fallback-{digest}"


def _configurable_string(config: RunnableConfig, key: str) -> str | None:
    value = config.get("configurable", {}).get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _as_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _truncate(text: str, max_chars: int = DESCRIPTION_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 1].rstrip()}…"


def _duration_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000
