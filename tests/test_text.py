"""Tests for ASCII text validation."""

from __future__ import annotations

from app.text import is_ascii_only


def test_is_ascii_only_accepts_english_place_name() -> None:
    assert is_ascii_only("Central Park") is True


def test_is_ascii_only_rejects_chinese() -> None:
    assert is_ascii_only("中央公园") is False


def test_is_ascii_only_rejects_non_ascii_accents() -> None:
    assert is_ascii_only("Café") is False


def test_is_ascii_only_accepts_empty_string() -> None:
    assert is_ascii_only("") is True
