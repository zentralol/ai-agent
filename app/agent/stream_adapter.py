"""Translate LangChain stream events into Zentra's public SSE contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    ToolMessage,
)

from app.schemas.events import (
    MessageDeltaEvent,
    RecommendationsEvent,
    StreamEvent,
    ToolFinishedEvent,
    ToolStartedEvent,
)
from app.schemas.recommendations import (
    CandidatePlace,
    RecommendationData,
    RecommendationItem,
)
from app.schemas.tools import ToolResponse, ToolStatus
from app.tools.attractions import GET_NEAREST_ATTRACTIONS_TOOL_NAME
from app.tools.itinerary import PLAN_ITINERARY_TOOL_NAME
from app.tools.places import GET_NEARBY_PLACES_TOOL_NAME
from app.tools.recommendations import SELECT_RECOMMENDED_PLACES_TOOL_NAME
from app.tools.recommendations_itinerary import RECOMMEND_TOOL_NAME
from app.target_time import (
    combine_anchor_date_and_stop_time,
    format_scheduled_at_display,
    normalize_target_time,
)


class LangChainStreamAdapter:
    """Stateful adapter from LangChain's stable v2 event stream to Zentra events.

    v2 is used deliberately over the newer v3 protocol: v3 is explicitly marked
    experimental by LangGraph and was found to corrupt streamed tool calls (name
    and id come back as empty strings) when reconstructing messages from chunks.
    """

    def __init__(self) -> None:
        self._usage_by_run_id: dict[str, dict[str, Any]] = {}
        self._streamed_text_run_ids: set[str] = set()
        self._candidates: dict[str, CandidatePlace] = {}
        self._recommendation_data: RecommendationData | None = None
        self._itinerary_anchor_time: str | None = None

    @property
    def usage(self) -> dict[str, Any] | None:
        """Aggregated model usage metadata reported by completed model runs."""

        return _combine_usage_metadata(self._usage_by_run_id)

    @property
    def recommendation_data(self) -> RecommendationData | None:
        """Return the last validated recommendation selection, if any."""

        return self._recommendation_data

    def recommendation_event(self) -> RecommendationsEvent | None:
        """Build one final structured event after the agent has completed."""

        if self._recommendation_data is None:
            return None
        return RecommendationsEvent(data=self._recommendation_data)

    def attach_recommendation_summary(self, summary: str) -> None:
        """Attach a natural-language plan summary to the current selection."""

        if self._recommendation_data is None:
            return
        self._recommendation_data = self._recommendation_data.model_copy(
            update={"summary": summary}
        )

    def infer_recommendations_from_text(self, assistant_text: str) -> None:
        """Backfill cards when the model skipped structured selection."""

        if self._recommendation_data is not None or not assistant_text.strip():
            return

        for source in ("recommend", "itinerary"):
            selected = self._match_candidates_in_text(assistant_text, source)
            if selected:
                target_time = (
                    self._itinerary_anchor_time if source == "itinerary" else None
                )
                self._recommendation_data = RecommendationData(
                    source=source,
                    items=selected,
                    target_time=target_time,
                )
                return

    def _match_candidates_in_text(
        self, assistant_text: str, source: Literal["recommend", "itinerary"]
    ) -> list[RecommendationItem]:
        candidates = [
            candidate for candidate in self._candidates.values() if candidate.source == source
        ]
        if not candidates:
            return []

        lowered = assistant_text.casefold()
        selected: list[RecommendationItem] = []
        for candidate in candidates:
            if candidate.name.casefold() not in lowered:
                continue
            selected.append(
                RecommendationItem(
                    **candidate.model_dump(),
                    rank=len(selected) + 1,
                    reason="",
                )
            )
        return selected

    def ui_parts(self) -> list[dict[str, Any]]:
        """Return persisted UI parts using the same snapshot sent to the client."""

        if self._recommendation_data is None:
            return []
        return [
            {
                "type": "data-places",
                "data": self._recommendation_data.model_dump(
                    mode="json", exclude_none=True
                ),
            }
        ]

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
            return self._handle_tool_end(raw_event)
        if event_name == "on_tool_error":
            return _tool_error_events(raw_event)
        return []

    def _handle_tool_end(self, raw_event: Mapping[object, object]) -> list[StreamEvent]:
        self._capture_itinerary_anchor_time(raw_event)
        events = _tool_finished_events(raw_event)
        for event in events:
            if not isinstance(event, ToolFinishedEvent):
                continue
            if event.result.status is not ToolStatus.SUCCESS:
                continue
            if event.tool_name in {
                GET_NEARBY_PLACES_TOOL_NAME,
                GET_NEAREST_ATTRACTIONS_TOOL_NAME,
                RECOMMEND_TOOL_NAME,
                PLAN_ITINERARY_TOOL_NAME,
            }:
                self._register_candidates(event)
                if event.tool_name == PLAN_ITINERARY_TOOL_NAME:
                    self._auto_select_itinerary_stops(event.result.data)
            elif event.tool_name == SELECT_RECOMMENDED_PLACES_TOOL_NAME:
                self._set_recommendations(event.result.data)
        return events


    def _capture_itinerary_anchor_time(self, raw_event: Mapping[object, object]) -> None:
        tool_name = _optional_string(raw_event.get("name"))
        if tool_name != PLAN_ITINERARY_TOOL_NAME:
            return

        data = _event_data(raw_event)
        if not isinstance(data, Mapping):
            return

        tool_input = data.get("input")
        if not isinstance(tool_input, Mapping):
            return

        anchor_time = _optional_string(tool_input.get("anchor_time"))
        if anchor_time is None:
            return

        try:
            self._itinerary_anchor_time = normalize_target_time(anchor_time)
        except ValueError:
            self._itinerary_anchor_time = None

    def _register_candidates(self, event: ToolFinishedEvent) -> None:
        if event.tool_name in {RECOMMEND_TOOL_NAME, PLAN_ITINERARY_TOOL_NAME}:
            collection_key = "candidates"
        elif event.tool_name == GET_NEARBY_PLACES_TOOL_NAME:
            collection_key = "places"
        else:
            collection_key = "attractions"
        raw_items = event.result.data.get(collection_key)
        if not isinstance(raw_items, list):
            return

        for raw in raw_items:
            if not isinstance(raw, Mapping):
                continue
            candidate = _candidate_from_tool_result(
                event.tool_name,
                raw,
                itinerary_anchor_time=self._itinerary_anchor_time,
            )
            if candidate is not None:
                self._candidates.setdefault(candidate.candidate_id, candidate)

    def _set_recommendations(self, data: Mapping[str, Any]) -> None:
        raw_items = data.get("recommendations")
        if not isinstance(raw_items, list):
            self._recommendation_data = None
            return

        selected: list[RecommendationItem] = []
        seen_ids: set[str] = set()
        for raw in raw_items:
            if not isinstance(raw, Mapping):
                continue
            candidate_id = _optional_string(raw.get("candidate_id"))
            if candidate_id is None or candidate_id in seen_ids:
                continue
            candidate = self._candidates.get(candidate_id)
            if candidate is None:
                continue
            reason = _optional_string(raw.get("reason")) or ""
            selected.append(
                RecommendationItem(
                    **candidate.model_dump(),
                    rank=len(selected) + 1,
                    reason=reason,
                )
            )
            seen_ids.add(candidate_id)

        if not selected:
            self._recommendation_data = None
            return

        sources = {item.source for item in selected}
        source = next(iter(sources)) if len(sources) == 1 else "mixed"
        self._recommendation_data = RecommendationData(source=source, items=selected)

    def _auto_select_itinerary_stops(self, data: Mapping[str, Any]) -> None:
        if self._recommendation_data is not None:
            return

        raw_stops = data.get("stops")
        if not isinstance(raw_stops, list):
            return

        selected: list[RecommendationItem] = []
        for raw in raw_stops:
            if not isinstance(raw, Mapping):
                continue
            candidate_id = _optional_string(raw.get("candidate_id"))
            if candidate_id is None:
                continue
            candidate = self._candidates.get(candidate_id)
            if candidate is None:
                continue
            reason = _as_string(raw.get("why_recommended"))[:500]
            selected.append(
                RecommendationItem(
                    **candidate.model_dump(),
                    rank=len(selected) + 1,
                    reason=reason,
                )
            )

        if not selected:
            return

        target_time = self._itinerary_anchor_time
        self._recommendation_data = RecommendationData(
            source="itinerary",
            items=selected,
            target_time=target_time,
        )

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


def _candidate_from_tool_result(
    tool_name: str,
    raw: Mapping[object, object],
    *,
    itinerary_anchor_time: str | None = None,
) -> CandidatePlace | None:
    candidate_id = _optional_string(raw.get("candidate_id"))
    name = _optional_string(raw.get("name"))
    lat = _finite_number(raw.get("lat"))
    lng = _finite_number(raw.get("lng"))
    if candidate_id is None or name is None or lat is None or lng is None:
        return None

    if tool_name == GET_NEARBY_PLACES_TOOL_NAME:
        subtitle = _as_string(raw.get("address"))
        detail = _join_detail(
            [
                _as_string(raw.get("primary_type")),
                _format_rating(raw.get("rating")),
                _format_distance(raw.get("distance_km")),
            ]
        )
        source: Literal["nearby", "attractions", "recommend", "itinerary"] = "nearby"
    elif tool_name == GET_NEAREST_ATTRACTIONS_TOOL_NAME:
        subtitle = _as_string(raw.get("neighborhood"))
        detail = _join_detail(
            [
                _as_string(raw.get("category")),
                _format_distance(raw.get("distance_km")),
            ]
        )
        source = "attractions"
    elif tool_name == PLAN_ITINERARY_TOOL_NAME:
        stop_time = _as_string(raw.get("time"))
        time_label = stop_time
        if itinerary_anchor_time and stop_time:
            scheduled_at = combine_anchor_date_and_stop_time(
                itinerary_anchor_time,
                stop_time,
            )
            if scheduled_at:
                time_label = format_scheduled_at_display(scheduled_at)
        subtitle = _join_detail(
            [
                time_label,
                _as_string(raw.get("neighborhood")),
            ]
        )
        detail = _join_detail(
            [
                _as_string(raw.get("category")),
                _as_string(raw.get("crowd_category")),
                _as_string(raw.get("hours")),
            ]
        )
        source = "itinerary"
    else:
        subtitle = _as_string(raw.get("neighborhood"))
        detail = _join_detail(
            [
                _as_string(raw.get("category")),
                _as_string(raw.get("crowd_category")),
                _as_string(raw.get("hours")),
            ]
        )
        source = "recommend"

    return CandidatePlace(
        candidate_id=candidate_id,
        source=source,
        name=name,
        lat=lat,
        lng=lng,
        subtitle=subtitle,
        detail=detail,
    )


def _format_rating(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"★ {value}"
    return ""


def _format_distance(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value} km"
    return ""


def _join_detail(parts: list[str]) -> str:
    return " · ".join(part for part in parts if part)


def _as_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _finite_number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if numeric == numeric and numeric not in {float("inf"), float("-inf")}:
            return numeric
    return None


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
