"""Request contracts for the agent chat/streaming endpoint."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ClientType(StrEnum):
    """Supported public clients that reach the agent through the gateway."""

    WEB = "web"
    IOS = "ios"


class PreferencesSnapshot(BaseModel):
    """Normalized user preferences supplied by the caller.

    Preferences are computed and normalized upstream; this service only consumes
    the snapshot and never mutates or re-derives it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    crowd_tolerance: str | None = Field(
        default=None,
        description="Relative tolerance for crowds, e.g. 'low', 'medium', 'high'.",
    )
    preferred_transport: str | None = Field(
        default=None,
        description="Preferred mode of transport, e.g. 'walk', 'transit', 'drive'.",
    )
    language: str | None = Field(
        default=None,
        description="Preferred response language as a BCP-47 tag, e.g. 'en', 'zh'.",
    )


class AgentStreamRequest(BaseModel):
    """Incoming request for the streaming chat endpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    user_id: str = Field(min_length=1, description="Resolved end-user identifier.")
    message: str = Field(min_length=1, description="Latest user message text.")
    client_type: ClientType = Field(description="Originating public client.")
    conversation_id: str | None = Field(
        default=None,
        description="Existing conversation to continue, if any.",
    )
    request_id: str | None = Field(
        default=None,
        description="Caller-provided correlation id for tracing.",
    )
    preferences: PreferencesSnapshot | None = Field(
        default=None,
        description="Normalized user preferences snapshot.",
    )
