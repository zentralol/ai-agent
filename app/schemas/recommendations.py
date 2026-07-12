"""Structured place recommendation contracts shared by tools and the stream."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PlaceSource = Literal["nearby", "attractions", "recommend", "itinerary"]
RecommendationSource = Literal["nearby", "attractions", "recommend", "itinerary", "mixed"]


class PlaceRecommendationSelection(BaseModel):
    """One candidate selected by the model through the selection tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(min_length=1)
    reason: str = Field(default="", max_length=500)


class CandidatePlace(BaseModel):
    """A navigable candidate returned by a place lookup tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(min_length=1)
    source: PlaceSource
    name: str = Field(min_length=1)
    lat: float
    lng: float
    subtitle: str = ""
    detail: str = ""


class RecommendationItem(CandidatePlace):
    """A validated candidate selected for the final card list."""

    rank: int = Field(ge=1)
    reason: str = Field(default="", max_length=500)


class RecommendationData(BaseModel):
    """Final ordered recommendation snapshot sent to and stored by clients."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: RecommendationSource
    items: list[RecommendationItem]
    # A short natural-language summary of the plan, generated from the tool
    # output after selection. Omitted when summarization is unavailable.
    summary: str | None = Field(default=None, max_length=1000)
    # Planned visit datetime (NY local ISO, e.g. 2026-07-10T16:00:00).
    target_time: str | None = Field(default=None, max_length=32)
