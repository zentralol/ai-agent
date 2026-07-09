"""Translate LangChain stream events into Zentra's public SSE contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import (
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
    """Stateful adapter from LangChain v3 stream events to Zentra events."""

    def __init__(self) -> None:
        self._tool_names_by_call_id: dict[str, str] = {}
        self._usage_by_message_id: dict[str, dict[str, Any]] = {}

    @property
    def usage(self) -> dict[str, Any] | None:
        """Aggregated model usage metadata reported by streamed messages."""

        return _combine_usage_metadata(self._usage_by_message_id)

    def to_zentra_events(self, raw_event: object) -> list[StreamEvent]:
        """Map one raw LangChain v3 event-stream payload to public events."""

        self._record_usage_metadata(raw_event)

        method = _event_method(raw_event)
        if method == "messages":
            return _message_events_from_data(_event_data(raw_event))
        if method == "tools":
            return self._tool_events_from_data(_event_data(raw_event))
        return []

    def _tool_events_from_data(self, data: object) -> list[StreamEvent]:
        if not isinstance(data, Mapping):
            return []

        event = data.get("event")
        tool_call_id = _optional_string(data.get("tool_call_id"))

        if event == "tool-started":
            tool_name = _optional_string(data.get("tool_name")) or "unknown_tool"
            self._remember_tool_name(tool_call_id, tool_name)
            return [
                ToolStartedEvent(tool_name=tool_name, tool_call_id=tool_call_id),
            ]

        if event == "tool-finished":
            output = data.get("output")
            tool_name = self._tool_name_for_finished_event(data, output, tool_call_id)
            self._remember_tool_name(tool_call_id, tool_name)
            return [
                ToolFinishedEvent(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    result=_coerce_tool_response(
                        _tool_output_content(output), tool_name
                    ),
                )
            ]

        if event == "tool-error":
            tool_name = self._known_tool_name(tool_call_id) or "unknown_tool"
            message = _optional_string(data.get("message")) or f"Tool failed: {tool_name}."
            return [
                ToolFinishedEvent(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    result=ToolResponse(
                        status=ToolStatus.ERROR,
                        summary=message,
                        next_actions=[
                            "Continue without this tool result or try again."
                        ],
                    ),
                )
            ]

        return []

    def _tool_name_for_finished_event(
        self,
        data: Mapping[object, object],
        output: object,
        tool_call_id: str | None,
    ) -> str:
        return (
            _optional_string(data.get("tool_name"))
            or _tool_message_name(output)
            or self._known_tool_name(tool_call_id)
            or "unknown_tool"
        )

    def _remember_tool_name(self, tool_call_id: str | None, tool_name: str) -> None:
        if tool_call_id is not None:
            self._tool_names_by_call_id[tool_call_id] = tool_name

    def _known_tool_name(self, tool_call_id: str | None) -> str | None:
        if tool_call_id is None:
            return None
        return self._tool_names_by_call_id.get(tool_call_id)

    def _record_usage_metadata(self, raw_event: object) -> None:
        if _event_method(raw_event) != "messages":
            return

        message = _message_from_data(_event_data(raw_event))
        if message is None:
            return

        usage = getattr(message, "usage_metadata", None)
        if not isinstance(usage, Mapping) or not usage:
            return

        message_id = _optional_string(getattr(message, "id", None)) or str(
            len(self._usage_by_message_id)
        )
        self._usage_by_message_id[message_id] = dict(usage)


def _event_method(raw_event: object) -> str | None:
    if not isinstance(raw_event, Mapping):
        return None
    method = raw_event.get("method")
    if not isinstance(method, str):
        return None
    return method


def _event_data(raw_event: object) -> object:
    if not isinstance(raw_event, Mapping):
        return None
    params = raw_event.get("params")
    if not isinstance(params, Mapping):
        return None
    return params.get("data")


def _message_events_from_data(data: object) -> list[StreamEvent]:
    message = _message_from_data(data)
    if message is None or isinstance(message, ToolMessage):
        return []

    text = _message_text(message)
    if not text:
        return []
    return [MessageDeltaEvent(text=text)]


def _message_from_data(data: object) -> BaseMessage | BaseMessageChunk | None:
    if not isinstance(data, tuple) or not data:
        return None
    message = data[0]
    if isinstance(message, BaseMessage | BaseMessageChunk):
        return message
    return None


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
    usage_by_message_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any] | None:
    if not usage_by_message_id:
        return None

    combined: dict[str, Any] = {}
    for usage in usage_by_message_id.values():
        for key, value in usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                previous = combined.get(key, 0)
                combined[key] = (
                    previous + value if isinstance(previous, int | float) else value
                )
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
