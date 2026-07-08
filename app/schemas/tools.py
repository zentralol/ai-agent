"""Tool response contract shared by all agent tools (see DEVELOPMENT_PLAN.md §6)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolStatus(StrEnum):
    """Outcome classification for a single tool invocation."""

    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class ToolResponse(BaseModel):
    """Uniform envelope every tool returns.

    Keeping the shape stable lets the agent reason about tool results without
    special-casing each tool.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: ToolStatus = Field(description="Outcome of the tool call.")
    summary: str = Field(description="One-line, human-readable result summary.")
    data: dict[str, Any] = Field(
        default_factory=dict, description="Typed payload produced by the tool."
    )
    next_actions: list[str] = Field(
        default_factory=list, description="Actionable follow-up suggestions."
    )
    artifacts: list[str] = Field(
        default_factory=list,
        description="Identifiers, paths, trace ids, or generated plan ids.",
    )
