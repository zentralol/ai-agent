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

from app.agent.loop import run_agent_loop
from app.llm import get_chat_model
from app.schemas.chat import AgentStreamRequest
from app.schemas.events import DoneEvent, MessageDeltaEvent, StreamEvent, WarningEvent
from app.tools.preferences import UserPreferenceTool, get_user_preference_tool
from app.tools.registry import build_tool_registry

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])

SSE_MEDIA_TYPE = "text/event-stream"

FALLBACK_DELTAS = (
    "The language model is not configured, ",
    "so this is a deterministic placeholder response. ",
    "Set LLM_API_KEY to enable real conversations.",
)


def _encode(event: StreamEvent) -> bytes:
    """Serialize a stream event as a single SSE ``data:`` frame."""

    payload = orjson.dumps(event.model_dump(mode="json"))
    return b"data: " + payload + b"\n\n"


async def _fallback_stream(request: AgentStreamRequest) -> AsyncIterator[bytes]:
    """Deterministic, dependency-free response used when no LLM is configured."""

    yield _encode(WarningEvent(message="LLM is not configured; using a placeholder reply."))
    for text in FALLBACK_DELTAS:
        yield _encode(MessageDeltaEvent(text=text))
    yield _encode(DoneEvent(conversation_id=request.conversation_id))


async def _event_stream(
    request: AgentStreamRequest,
    model: BaseChatModel | None,
    preference_tool: UserPreferenceTool,
) -> AsyncIterator[bytes]:
    if model is None:
        async for frame in _fallback_stream(request):
            yield frame
        return

    registry = build_tool_registry(preference_tool)
    async for event in run_agent_loop(request, model, registry):
        yield _encode(event)


_ModelDependency = Depends(get_chat_model)
_PreferenceToolDependency = Depends(get_user_preference_tool)


@router.post("/stream")
async def agent_stream(
    request: AgentStreamRequest,
    model: BaseChatModel | None = _ModelDependency,
    preference_tool: UserPreferenceTool = _PreferenceToolDependency,
) -> StreamingResponse:
    """Stream typed chat events for a single user message."""

    return StreamingResponse(
        _event_stream(request, model, preference_tool), media_type=SSE_MEDIA_TYPE
    )
