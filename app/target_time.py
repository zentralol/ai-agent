"""NY-local naive datetime helpers for planned visit times."""

from __future__ import annotations

import re
from datetime import datetime

# Full date + time, no timezone offset: YYYY-MM-DDTHH:mm or YYYY-MM-DDTHH:mm:ss
TARGET_TIME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$")
_STOP_TIME_PATTERN = re.compile(r"^\d{1,2}:\d{2}$")


def normalize_target_time(value: str | None) -> str | None:
    """Validate and normalize a planned visit datetime for storage/streaming."""

    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if not TARGET_TIME_PATTERN.fullmatch(stripped):
        msg = "target_time must be NY local ISO datetime YYYY-MM-DDTHH:mm:ss"
        raise ValueError(msg)
    if len(stripped) == 16:
        return f"{stripped}:00"
    return stripped


def combine_anchor_date_and_stop_time(
    anchor_time: str,
    stop_time: str,
) -> str | None:
    """Merge anchor date with a stop's HH:MM clock time into one ISO datetime."""

    normalized_anchor = normalize_target_time(anchor_time)
    if normalized_anchor is None:
        return None

    stripped_stop = stop_time.strip()
    if not _STOP_TIME_PATTERN.fullmatch(stripped_stop):
        return None

    hour, minute = stripped_stop.split(":", 1)
    return f"{normalized_anchor[:10]}T{hour.zfill(2)}:{minute}:00"


def format_scheduled_at_display(iso: str) -> str:
    """Format a stored NY-local ISO datetime for card subtitles."""

    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})", iso.strip())
    if not match:
        return iso

    year, month, day, hour, minute = match.groups()
    date = datetime(
        int(year),
        int(month),
        int(day),
        int(hour),
        int(minute),
    )
    month = date.strftime("%b")
    day = str(date.day)
    hour = date.strftime("%I").lstrip("0") or "12"
    minute = date.strftime("%M")
    ampm = date.strftime("%p")
    return f"{month} {day}, {date.year}, {hour}:{minute} {ampm}"
