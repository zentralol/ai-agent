"""Schemas for controlled user preference lookup."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PreferenceCategory(StrEnum):
    """Narrow preference groups the agent is allowed to request."""

    TRAVEL_STYLE = "travel_style"
    CROWD = "crowd"
    TRANSPORT = "transport"
    BUDGET = "budget"
    ACCESSIBILITY = "accessibility"
    LANGUAGE = "language"
    INTERESTS = "interests"


class UserPreferences(BaseModel):
    """Sanitized preference snapshot returned by the backend-controlled tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    travel_style: str | None = Field(default=None)
    crowd_tolerance: str | None = Field(default=None)
    preferred_transport: str | None = Field(default=None)
    budget: str | None = Field(default=None)
    accessibility: list[str] = Field(default_factory=list)
    language: str | None = Field(default=None)
    interests: list[str] = Field(default_factory=list)


CATEGORY_FIELDS: dict[PreferenceCategory, tuple[str, ...]] = {
    PreferenceCategory.TRAVEL_STYLE: ("travel_style",),
    PreferenceCategory.CROWD: ("crowd_tolerance",),
    PreferenceCategory.TRANSPORT: ("preferred_transport",),
    PreferenceCategory.BUDGET: ("budget",),
    PreferenceCategory.ACCESSIBILITY: ("accessibility",),
    PreferenceCategory.LANGUAGE: ("language",),
    PreferenceCategory.INTERESTS: ("interests",),
}


def dump_selected_preferences(
    preferences: UserPreferences, categories: tuple[PreferenceCategory, ...]
) -> dict[str, Any]:
    """Return only fields covered by the requested categories."""

    raw = preferences.model_dump(mode="json")
    selected_fields = {
        field for category in categories for field in CATEGORY_FIELDS[category]
    }
    return {
        key: value
        for key, value in raw.items()
        if key in selected_fields and value not in (None, [], {})
    }
