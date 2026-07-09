"""Translate LangChain stream events into Zentra's public SSE contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    ToolMessage,
)

from app.schemas.events import (
    MessageDeltaEvent,
    StreamEvent,
    ToolFinishedEvent,
    ToolStartedEvent,
)
from app.schemas.tools import ToolResponse, ToolStatus


class LangChainStreamAdapter:
    """Stateful adapter from LangChain's stable v2 event stream to Zentra events.

    v2 is used deliberately over the newer v3 protocol: v3 is explicitly marked
    experimental by LangGraph and was found to corrupt streamed tool calls (name
    and id come back as empty strings) when reconstructing messages from chunks.
    """

    def __init__(self) -> None:
        self._usage_by_run_id: dict[str, dict[str, Any]] = {}
        self._streamed_text_run_ids: set[str] = set()

    @property
    def usage(self) -> dict[str, Any] | None:
        """Aggregated model usage metadata reported by completed model runs."""

        return _combine_usage_metadata(self._usage_by_run_id)

    def to_zentra_events(self, raw_event: object) -> list[StreamEvent]:
        """Map one raw LangChain v2 event-stream payload to public events."""

        if not isinstance(raw_event, Mapping):
            return []

        event_name = raw_event.get("event")
        if event_name == "on_chat_model_stream":
            return self._handle_message_stream(raw_event)
        if event_name == "on_chat_model_end":
            return self._handle_message_end(raw_event)
        if event_name == "on_tool_start":
            return _tool_started_events(raw_event)
        if event_name == "on_tool_end":
            return _tool_finished_events(raw_event)
        if event_name == "on_tool_error":
            return _tool_error_events(raw_event)
        return []

    def _handle_message_stream(self, raw_event: Mapping[object, object]) -> list[StreamEvent]:
        events = _message_events(raw_event)
        if events:
            run_id = _optional_string(raw_event.get("run_id"))
            if run_id is not None:
                self._streamed_text_run_ids.add(run_id)
        return events

    def _handle_message_end(self, raw_event: Mapping[object, object]) -> list[StreamEvent]:
        """Handle a completed model run: record usage and, for models that never
        emitted token-level chunks (test doubles, non-streaming providers), emit
        the full response as a single fallback delta.
        """

        data = _event_data(raw_event)
        output = data.get("output") if isinstance(data, Mapping) else None
        if not isinstance(output, AIMessage):
            return []

        self._record_usage(raw_event, output)

        run_id = _optional_string(raw_event.get("run_id"))
        if run_id is not None and run_id in self._streamed_text_run_ids:
            return []

        text = _message_text(output)
        if not text:
            return []
        return [MessageDeltaEvent(text=text)]

    def _record_usage(self, raw_event: Mapping[object, object], output: AIMessage) -> None:
        usage = output.usage_metadata
        if not usage:
            return

        run_id = _optional_string(raw_event.get("run_id")) or str(len(self._usage_by_run_id))
        self._usage_by_run_id[run_id] = dict(usage)


def _event_data(raw_event: Mapping[object, object]) -> object:
    return raw_event.get("data")


def _message_events(raw_event: Mapping[object, object]) -> list[StreamEvent]:
    data = _event_data(raw_event)
    chunk = data.get("chunk") if isinstance(data, Mapping) else None
    if not isinstance(chunk, BaseMessage | BaseMessageChunk) or isinstance(chunk, ToolMessage):
        return []

    text = _message_text(chunk)
    if not text:
        return []
    return [MessageDeltaEvent(text=text)]


def _tool_started_events(raw_event: Mapping[object, object]) -> list[StreamEvent]:
    tool_name = _optional_string(raw_event.get("name")) or "unknown_tool"
    return [ToolStartedEvent(tool_name=tool_name, tool_call_id=None)]


def _tool_finished_events(raw_event: Mapping[object, object]) -> list[StreamEvent]:
    data = _event_data(raw_event)
    output = data.get("output") if isinstance(data, Mapping) else None
    tool_name = (
        _tool_message_name(output) or _optional_string(raw_event.get("name")) or "unknown_tool"
    )
    tool_call_id = output.tool_call_id if isinstance(output, ToolMessage) else None
    return [
        ToolFinishedEvent(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            result=_coerce_tool_response(_tool_output_content(output), tool_name),
        )
    ]


def _tool_error_events(raw_event: Mapping[object, object]) -> list[StreamEvent]:
    data = _event_data(raw_event)
    if not isinstance(data, Mapping):
        data = {}

    tool_name = _optional_string(raw_event.get("name")) or "unknown_tool"
    tool_call_id = _optional_string(data.get("tool_call_id"))
    error = data.get("error")
    message = (_optional_string(str(error)) if error is not None else None) or (
        f"Tool failed: {tool_name}."
    )
    return [
        ToolFinishedEvent(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            result=ToolResponse(
                status=ToolStatus.ERROR,
                summary=message,
                next_actions=["Continue without this tool result or try again."],
            ),
        )
    ]


def _message_text(message: BaseMessage | BaseMessageChunk) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text

    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _tool_message_name(output: object) -> str | None:
    if not isinstance(output, ToolMessage):
        return None
    return _optional_string(output.name)


def _tool_output_content(output: object) -> object:
    if isinstance(output, ToolMessage):
        return output.content
    return output


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _combine_usage_metadata(
    usage_by_run_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not usage_by_run_id:
        return None

    combined: dict[str, Any] = {}
    for usage in usage_by_run_id.values():
        for key, value in usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                previous = combined.get(key, 0)
                combined[key] = previous + value if isinstance(previous, int | float) else value
                continue
            combined.setdefault(key, value)

    return combined or None


def _coerce_tool_response(content: object, tool_name: str) -> ToolResponse:
    """Parse a tool's raw content into the public response envelope."""

    if isinstance(content, str):
        try:
            return ToolResponse.model_validate_json(content)
        except ValueError:
            return ToolResponse(
                status=ToolStatus.SUCCESS,
                summary=f"Tool returned text: {tool_name}.",
                data={"content": content},
            )
    return ToolResponse(
        status=ToolStatus.SUCCESS,
        summary=f"Tool returned a result: {tool_name}.",
        data={"content": str(content)},
    )
