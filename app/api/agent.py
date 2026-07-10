"""Chat/streaming endpoint for the agent service.

Validates the incoming request and streams typed :mod:`app.schemas.events` events
as Server-Sent Events (SSE). When an LLM is configured it streams the model's
tokens; otherwise it falls back to a deterministic response so the contract still
works with zero external dependencies.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import orjson
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.language_models.chat_models import BaseChatModel

from app.agent.runner import run_agent_stream
from app.api.internal_auth import require_internal_auth
from app.llm import get_chat_model
from app.schemas.chat import AgentStreamRequest
from app.schemas.events import DoneEvent, MessageDeltaEvent, StreamEvent, WarningEvent
from app.tools.catalog import AGENT_TOOLS

router = APIRouter(
    prefix="/api/v1/agent",
    tags=["agent"],
    dependencies=[Depends(require_internal_auth)],
)

SSE_MEDIA_TYPE = "text/event-stream"

FALLBACK_DELTAS = (
    "The language model is not configured, ",
    "so this is a deterministic placeholder response. ",
    "Set LLM_API_KEY to enable real conversations.",
)


def _encode(event: StreamEvent) -> bytes:
    """Serialize a stream event as a single SSE ``data:`` frame."""

    payload = orjson.dumps(event.model_dump(mode="json", exclude_none=True))
    return b"data: " + payload + b"\n\n"


def _with_sequence(event: StreamEvent, sequence: int) -> StreamEvent:
    """Attach the public sequence number at the API boundary."""

    return event.model_copy(update={"sequence": sequence})


async def _fallback_events(request: AgentStreamRequest) -> AsyncIterator[StreamEvent]:
    """Deterministic, dependency-free response used when no LLM is configured."""

    yield WarningEvent(message="LLM is not configured; using a placeholder reply.")
    for text in FALLBACK_DELTAS:
        yield MessageDeltaEvent(text=text)
    yield DoneEvent(conversation_id=request.conversation_id)


async def _event_stream(
    request: AgentStreamRequest,
    model: BaseChatModel | None,
) -> AsyncIterator[bytes]:
    sequence = 1

    if model is None:
        event_source = _fallback_events(request)
    else:
        event_source = run_agent_stream(request, model, AGENT_TOOLS)

    async for event in event_source:
        yield _encode(_with_sequence(event, sequence))
        sequence += 1


_ModelDependency = Depends(get_chat_model)


@router.post("/stream")
async def agent_stream(
    request: AgentStreamRequest,
    model: BaseChatModel | None = _ModelDependency,
) -> StreamingResponse:
    """Stream typed chat events for a single user message."""

    return StreamingResponse(
        _event_stream(request, model), media_type=SSE_MEDIA_TYPE
    )
