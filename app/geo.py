"""Geo helpers shared by location-aware tools."""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

from langchain_core.runnables import RunnableConfig

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance between two points in kilometers."""

    lat1_r, lng1_r, lat2_r, lng2_r = map(radians, (lat1, lng1, lat2, lng2))
    d_lat = lat2_r - lat1_r
    d_lng = lng2_r - lng1_r
    a = sin(d_lat / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(d_lng / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(a))


def as_float(value: object) -> float | None:
    """Coerce a value to float, rejecting booleans and non-numerics."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def configurable_float(config: RunnableConfig, key: str) -> float | None:
    """Read a float from the tool runtime config's ``configurable`` map."""

    return as_float(config.get("configurable", {}).get(key))
