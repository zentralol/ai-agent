"""Text validation helpers shared by agent tools."""


def is_ascii_only(text: str) -> bool:
    """Return True if every character has ord(c) < 128 (no CJK / non-ASCII)."""

    for char in text:
        if ord(char) >= 128:
            return False
    return True
