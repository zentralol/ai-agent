"""Tests for the conversation title heuristic (ported from the frontend)."""

from __future__ import annotations

from app.conversations.title import title_from_user_message


def test_short_message_is_used_verbatim() -> None:
    assert title_from_user_message("Plan my day") == "Plan my day"


def test_collapses_whitespace() -> None:
    assert title_from_user_message("  Plan   my    day  ") == "Plan my day"


def test_limits_to_six_words() -> None:
    assert (
        title_from_user_message("one two three four five six seven eight")
        == "one two three four five six"
    )


def test_truncates_long_single_run_with_ellipsis() -> None:
    result = title_from_user_message("x" * 80)
    assert result.endswith("…")
    assert len(result) <= 50


def test_empty_message_returns_empty() -> None:
    assert title_from_user_message("   ") == ""
