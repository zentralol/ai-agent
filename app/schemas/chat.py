"""Request contracts for the agent chat/streaming endpoint."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ClientType(StrEnum):
    """Supported public clients that reach the agent through the gateway."""

    WEB = "web"
    IOS = "ios"


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
