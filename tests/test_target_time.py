"""Tests for NY-local target time helpers."""

from __future__ import annotations

import pytest

from app.target_time import (
    combine_anchor_date_and_stop_time,
    format_scheduled_at_display,
    normalize_target_time,
)


def test_combine_anchor_date_and_stop_time() -> None:
    assert (
        combine_anchor_date_and_stop_time("2026-07-06T10:00:00", "16:00")
        == "2026-07-06T16:00:00"
    )


def test_format_scheduled_at_display_includes_date_and_time() -> None:
    formatted = format_scheduled_at_display("2026-07-06T16:00:00")
    assert "Jul" in formatted
    assert "2026" in formatted
    assert "4:00 PM" in formatted


def test_normalize_target_time_rejects_time_only() -> None:
    with pytest.raises(ValueError):
        normalize_target_time("16:00")

def test_normalize_target_time_strips_timezone_offset() -> None:
    assert normalize_target_time("2026-07-10T16:00:00-04:00") == "2026-07-10T16:00:00"
    assert normalize_target_time("2026-07-10T16:00:00Z") == "2026-07-10T16:00:00"
