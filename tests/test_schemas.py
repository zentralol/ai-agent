"""Schema validation tests covering Phase 1 acceptance criteria."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from app.schemas.chat import AgentStreamRequest, ClientType
from app.schemas.events import (
    DoneEvent,
    ErrorEvent,
    EventType,
    MessageDeltaEvent,
    StreamEvent,
    ToolFinishedEvent,
)
from app.schemas.preferences import PreferenceCategory, UserPreferences, dump_selected_preferences
from app.schemas.tools import ToolResponse, ToolStatus

_stream_event_adapter: TypeAdapter[StreamEvent] = TypeAdapter(StreamEvent)


def test_valid_request_parses() -> None:
    # Arrange
    payload = {
        "user_id": "user-1",
        "message": "Where is quiet nearby?",
        "client_type": "web",
    }

    # Act
    request = AgentStreamRequest.model_validate(payload)

    # Assert
    assert request.client_type is ClientType.WEB


def test_missing_user_context_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentStreamRequest.model_validate(
            {"message": "hi", "client_type": "web"}
        )


def test_empty_message_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentStreamRequest.model_validate(
            {"user_id": "u1", "message": "", "client_type": "web"}
        )


def test_unsupported_client_type_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentStreamRequest.model_validate(
            {"user_id": "u1", "message": "hi", "client_type": "android"}
        )


def test_inline_preferences_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentStreamRequest.model_validate(
            {
                "user_id": "u1",
                "message": "hi",
                "client_type": "web",
                "preferences": {"crowd_tolerance": "low"},
            }
        )


def test_forbidden_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentStreamRequest.model_validate(
            {
                "user_id": "u1",
                "message": "hi",
                "client_type": "web",
                "unexpected": "value",
            }
        )


def test_request_is_immutable() -> None:
    request = AgentStreamRequest(user_id="u1", message="hi", client_type=ClientType.WEB)
    with pytest.raises(ValidationError):
        request.user_id = "u2"


def test_selected_preferences_only_returns_requested_categories() -> None:
    preferences = UserPreferences(
        crowd_tolerance="low",
        preferred_transport="walk",
        language="zh",
        interests=["parks"],
    )

    selected = dump_selected_preferences(
        preferences,
        (PreferenceCategory.CROWD, PreferenceCategory.LANGUAGE),
    )

    assert selected == {"crowd_tolerance": "low", "language": "zh"}


@pytest.mark.parametrize(
    "event",
    [
        MessageDeltaEvent(text="hello"),
        DoneEvent(conversation_id="c1"),
        ErrorEvent(code="E_TIMEOUT", message="timed out"),
        ToolFinishedEvent(
            tool_name="predict_crowd_batch",
            result=ToolResponse(status=ToolStatus.SUCCESS, summary="ok"),
        ),
    ],
)
def test_stream_event_roundtrip(event: StreamEvent) -> None:
    # Act: dump then re-parse through the discriminated union.
    dumped = event.model_dump(mode="json")
    parsed = _stream_event_adapter.validate_python(dumped)

    # Assert
    assert parsed == event


def test_invalid_event_type_rejected() -> None:
    with pytest.raises(ValidationError):
        _stream_event_adapter.validate_python({"type": "nonsense", "text": "x"})


def test_event_missing_required_field_rejected() -> None:
    # message_delta requires `text`.
    with pytest.raises(ValidationError):
        _stream_event_adapter.validate_python({"type": EventType.MESSAGE_DELTA.value})
