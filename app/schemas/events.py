"""Stream event contracts for the agent chat endpoint (see DEVELOPMENT_PLAN.md §5).

Events are a discriminated union keyed on ``type`` so clients can switch on the
event kind without parsing prose.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

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

    sequence: int | None = Field(
        default=None,
        ge=1,
        description="Monotonic SSE sequence number assigned by the API edge.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional non-contractual metadata for diagnostics or UI hints.",
    )


class MessageDeltaEvent(_BaseEvent):
    """Incremental chunk of assistant text.

    When the configured model supports streaming this is a token/text delta. Test
    models and non-streaming providers may emit a complete message in one chunk.
    """

    type: Literal[EventType.MESSAGE_DELTA] = EventType.MESSAGE_DELTA
    text: str = Field(description="Text fragment to append to the response.")


class ToolStartedEvent(_BaseEvent):
    """Signals a tool invocation has begun."""

    type: Literal[EventType.TOOL_STARTED] = EventType.TOOL_STARTED
    tool_name: str = Field(description="Name of the tool being called.")
    tool_call_id: str | None = Field(
        default=None,
        description="Provider/runtime tool call identifier, when available.",
    )


class ToolFinishedEvent(_BaseEvent):
    """Signals a tool invocation has completed."""

    type: Literal[EventType.TOOL_FINISHED] = EventType.TOOL_FINISHED
    tool_name: str = Field(description="Name of the tool that finished.")
    tool_call_id: str | None = Field(
        default=None,
        description="Provider/runtime tool call identifier, when available.",
    )
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
    usage: dict[str, Any] | None = Field(
        default=None,
        description="Aggregated model usage metadata when the provider reports it.",
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
