"""Stream event contracts for the agent chat endpoint (see DEVELOPMENT_PLAN.md §5).

Events are a discriminated union keyed on ``type`` so clients can switch on the
event kind without parsing prose.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.tools import ToolResponse


class EventType(StrEnum):
    """Enumeration of every stream event kind."""

    MESSAGE_DELTA = "message_delta"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    BACKEND_CAPABILITY_RESULT = "backend_capability_result"
    WARNING = "warning"
    DONE = "done"
    ERROR = "error"


class _BaseEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MessageDeltaEvent(_BaseEvent):
    """Incremental chunk of assistant text."""

    type: Literal[EventType.MESSAGE_DELTA] = EventType.MESSAGE_DELTA
    text: str = Field(description="Text fragment to append to the response.")


class ToolStartedEvent(_BaseEvent):
    """Signals a tool invocation has begun."""

    type: Literal[EventType.TOOL_STARTED] = EventType.TOOL_STARTED
    tool_name: str = Field(description="Name of the tool being called.")


class ToolFinishedEvent(_BaseEvent):
    """Signals a tool invocation has completed."""

    type: Literal[EventType.TOOL_FINISHED] = EventType.TOOL_FINISHED
    tool_name: str = Field(description="Name of the tool that finished.")
    result: ToolResponse = Field(description="Structured tool result envelope.")


class BackendCapabilityResultEvent(_BaseEvent):
    """Carries a backend-owned capability result surfaced to the client."""

    type: Literal[EventType.BACKEND_CAPABILITY_RESULT] = (
        EventType.BACKEND_CAPABILITY_RESULT
    )
    capability: str = Field(description="Backend capability identifier.")
    result: ToolResponse = Field(description="Structured capability result envelope.")


class WarningEvent(_BaseEvent):
    """Non-fatal warning surfaced during the run."""

    type: Literal[EventType.WARNING] = EventType.WARNING
    message: str = Field(description="Human-readable warning message.")


class DoneEvent(_BaseEvent):
    """Terminal event marking a successful stream completion."""

    type: Literal[EventType.DONE] = EventType.DONE
    conversation_id: str | None = Field(
        default=None, description="Conversation id associated with this run."
    )


class ErrorEvent(_BaseEvent):
    """Terminal event marking a recoverable stream failure."""

    type: Literal[EventType.ERROR] = EventType.ERROR
    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human-readable error message.")


StreamEvent = Annotated[
    MessageDeltaEvent
    | ToolStartedEvent
    | ToolFinishedEvent
    | BackendCapabilityResultEvent
    | WarningEvent
    | DoneEvent
    | ErrorEvent,
    Field(discriminator="type"),
]

EVENT_TYPES: frozenset[EventType] = frozenset(EventType)
