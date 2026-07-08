"""Chat/streaming endpoint for the agent service.

Validates the incoming request and streams typed :mod:`app.schemas.events` events
as Server-Sent Events (SSE). When an LLM is configured it streams the model's
tokens; otherwise it falls back to a deterministic response so the contract still
works with zero external dependencies.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import orjson
import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_chat_model
from app.schemas.chat import AgentStreamRequest, PreferencesSnapshot
from app.schemas.events import DoneEvent, ErrorEvent, MessageDeltaEvent, StreamEvent, WarningEvent

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])

logger = structlog.get_logger(__name__)

SSE_MEDIA_TYPE = "text/event-stream"

SYSTEM_PROMPT = (
    "You are Zentra's travel assistant. You help users find less crowded places, "
    "plan routes, and answer travel questions. Be concise, friendly, and practical. "
    "Only state facts you are confident about; if you lack data, say so."
)

FALLBACK_DELTAS = (
    "The language model is not configured, ",
    "so this is a deterministic placeholder response. ",
    "Set LLM_API_KEY to enable real conversations.",
)


def _encode(event: StreamEvent) -> bytes:
    """Serialize a stream event as a single SSE ``data:`` frame."""

    payload = orjson.dumps(event.model_dump(mode="json"))
    return b"data: " + payload + b"\n\n"


def _preferences_hint(preferences: PreferencesSnapshot | None) -> str:
    """Render a compact preferences hint for the system prompt."""

    if preferences is None:
        return ""

    parts = [
        f"{label}: {value}"
        for label, value in (
            ("crowd tolerance", preferences.crowd_tolerance),
            ("preferred transport", preferences.preferred_transport),
            ("language", preferences.language),
        )
        if value
    ]
    if not parts:
        return ""
    return "\n\nUser preferences — " + "; ".join(parts) + "."


def _chunk_text(content: object) -> str:
    """Extract plain text from a LangChain message chunk's content."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


async def _fallback_stream(request: AgentStreamRequest) -> AsyncIterator[bytes]:
    """Deterministic, dependency-free response used when no LLM is configured."""

    yield _encode(WarningEvent(message="LLM is not configured; using a placeholder reply."))
    for text in FALLBACK_DELTAS:
        yield _encode(MessageDeltaEvent(text=text))
    yield _encode(DoneEvent(conversation_id=request.conversation_id))


async def _llm_stream(
    request: AgentStreamRequest, model: BaseChatModel
) -> AsyncIterator[bytes]:
    """Stream the LLM's tokens as message_delta events, ending with done/error."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT + _preferences_hint(request.preferences)),
        HumanMessage(content=request.message),
    ]
    try:
        async for chunk in model.astream(messages):
            text = _chunk_text(chunk.content)
            if text:
                yield _encode(MessageDeltaEvent(text=text))
    except Exception:
        logger.exception("llm_stream_failed", user_id=request.user_id)
        yield _encode(
            ErrorEvent(
                code="LLM_ERROR",
                message="The assistant failed to generate a response. Please try again.",
            )
        )
        return

    yield _encode(DoneEvent(conversation_id=request.conversation_id))


async def _event_stream(
    request: AgentStreamRequest, model: BaseChatModel | None
) -> AsyncIterator[bytes]:
    if model is None:
        async for frame in _fallback_stream(request):
            yield frame
        return

    async for frame in _llm_stream(request, model):
        yield frame


_ModelDependency = Depends(get_chat_model)


@router.post("/stream")
async def agent_stream(
    request: AgentStreamRequest,
    model: BaseChatModel | None = _ModelDependency,
) -> StreamingResponse:
    """Stream typed chat events for a single user message."""

    return StreamingResponse(_event_stream(request, model), media_type=SSE_MEDIA_TYPE)
