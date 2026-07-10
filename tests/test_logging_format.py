"""Tests for ANSI-colored console logging."""

from __future__ import annotations

import logging

import pytest

from app.logging_format import (
    ColoredLogFormatter,
    colorize_message,
    use_color,
)


@pytest.mark.parametrize(
    ("env", "isatty", "expected"),
    [
        ({"NO_COLOR": "1"}, True, False),
        ({}, False, False),
        ({}, True, True),
        ({"FORCE_COLOR": "1"}, False, True),
        ({"LOG_COLOR": "1"}, False, True),
        ({"FORCE_COLOR": "0"}, False, False),
        ({"NO_COLOR": "1", "FORCE_COLOR": "1"}, True, False),
    ],
)
def test_use_color_respects_no_color_and_tty(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    isatty: bool,
    expected: bool,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("LOG_COLOR", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("app.logging_format.sys.stdout.isatty", lambda: isatty)
    assert use_color() is expected


def test_colorize_message_disabled_returns_plain_text() -> None:
    message = (
        "tool_call_start tool=get_nearby_places request_id=r1 "
        "conversation_id=c1 duration_ms=12.34 status=success"
    )
    assert colorize_message(message, enabled=False) == message


def test_colorize_message_enabled_adds_ansi_codes() -> None:
    message = (
        "tool_call_start tool=get_nearby_places status=success "
        "duration_ms=12.34 request_id=r1"
    )
    colored = colorize_message(message, enabled=True)
    assert colored != message
    assert "\033[" in colored
    assert "tool_call_start" in colored
    assert "get_nearby_places" in colored


def test_colorize_message_styles_structlog_event_key() -> None:
    message = "user_id='u1' event='agent_tool_step_limit_reached' level='warning'"
    colored = colorize_message(message, enabled=True)
    assert "\033[" in colored
    assert "agent_tool_step_limit_reached" in colored


def test_colored_log_formatter_smoke() -> None:
    formatter = ColoredLogFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    record = logging.LogRecord(
        name="zentra_agent.tools.places",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=(
            "tool_call_end tool=%s status=%s request_id=%s "
            "conversation_id=%s duration_ms=%.2f"
        ),
        args=("get_nearby_places", "success", "r1", "c1", 42.5),
        exc_info=None,
    )

    plain = formatter.format(record)
    assert "tool_call_end" in plain
    assert "get_nearby_places" in plain
    assert "42.50" in plain

    colored = colorize_message(record.getMessage(), enabled=True)
    assert "\033[" in colored


def test_colored_log_formatter_no_color_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    formatter = ColoredLogFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    record = logging.LogRecord(
        name="zentra_agent.tools.crowd",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="crowd_tool_unconfigured missing=BACKEND_API_BASE_URL",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    assert "\033[" not in output
    assert "crowd_tool_unconfigured" in output


def test_colored_log_formatter_force_color_without_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setattr("app.logging_format.sys.stdout.isatty", lambda: False)

    formatter = ColoredLogFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    record = logging.LogRecord(
        name="zentra_agent.tools.places",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="tool_call_start tool=%s request_id=%s",
        args=("get_nearby_places", "r1"),
        exc_info=None,
    )
    output = formatter.format(record)
    assert "\033[" in output
    assert "tool_call_start" in output
