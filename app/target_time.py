"""NY-local naive datetime helpers for planned visit times."""

from __future__ import annotations

import re

# Full date + time, no timezone offset: YYYY-MM-DDTHH:mm or YYYY-MM-DDTHH:mm:ss
TARGET_TIME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$")


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
