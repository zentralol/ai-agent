"""Schemas for controlled user preference lookup."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class UserPreferences(BaseModel):
    """Sanitized preference snapshot returned by the backend-controlled tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    travel_pace: str | None = Field(default=None)
    crowd_tolerance: str | None = Field(default=None)
    budget_range: str | None = Field(default=None)
    interests: list[str] = Field(default_factory=list)
    mobility_needs: list[str] = Field(default_factory=list)
    dietary_needs: list[str] = Field(default_factory=list)
    inclusion_needs: list[str] = Field(default_factory=list)
    onboarding_completed: bool = Field(default=False)
