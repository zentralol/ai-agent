"""Chat/streaming endpoint for the agent service.

This is the minimal contract layer: it validates the incoming request and emits a
deterministic Server-Sent Events (SSE) stream of typed :mod:`app.schemas.events`
events. There is no LLM, tool, or backend call yet — later phases replace the
deterministic body with the LangGraph conversation workflow.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import orjson
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.schemas.chat import AgentStreamRequest
from app.schemas.events import DoneEvent, MessageDeltaEvent, StreamEvent

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])

SSE_MEDIA_TYPE = "text/event-stream"


def _encode(event: StreamEvent) -> bytes:
    """Serialize a stream event as a single SSE ``data:`` frame."""

    payload = orjson.dumps(event.model_dump(mode="json"))
    return b"data: " + payload + b"\n\n"


async def _event_stream(request: AgentStreamRequest) -> AsyncIterator[bytes]:
    """Yield a deterministic sequence of stream events for the request.

    The body acknowledges the message without invoking any model or backend, so
    the contract can be exercised end to end with zero external dependencies.
    """

    deltas = (
        "Received your message. ",
        "The agent workflow is not wired up yet, ",
        "so this is a deterministic placeholder response.",
    )
    for text in deltas:
        yield _encode(MessageDeltaEvent(text=text))

    yield _encode(DoneEvent(conversation_id=request.conversation_id))


@router.post("/stream")
async def agent_stream(request: AgentStreamRequest) -> StreamingResponse:
    """Stream typed chat events for a single user message."""

    return StreamingResponse(_event_stream(request), media_type=SSE_MEDIA_TYPE)
