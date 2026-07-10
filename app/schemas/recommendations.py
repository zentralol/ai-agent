"""Structured place recommendation contracts shared by tools and the stream."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PlaceSource = Literal["nearby", "attractions"]
RecommendationSource = Literal["nearby", "attractions", "mixed"]


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
