"""Derive a conversation title from the first user message.

Ported from the frontend heuristic in my-app ``lib/assistant/titleUtils.ts`` so
titles stay consistent regardless of which client started the conversation.
This intentionally avoids an extra LLM call.
"""

from __future__ import annotations

import re

TITLE_MAX_LENGTH = 50
TITLE_MAX_WORDS = 6

_WHITESPACE = re.compile(r"\s+")


def _limit_word_count(text: str, max_words: int) -> str:
    words = [word for word in text.split(" ") if word]
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _truncate_to_last_complete_word(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text

    prefix = text[:max_length]
    if max_length < len(text) and text[max_length].isspace():
        return prefix.rstrip()

    last_space_index = prefix.rfind(" ")
    if last_space_index > 0:
        return f"{prefix[:last_space_index].rstrip()}…"

    return f"{text[: max_length - 1]}…"


def title_from_user_message(text: str, max_length: int = TITLE_MAX_LENGTH) -> str:
    """Build a compact title from a user message."""

    normalized = _WHITESPACE.sub(" ", text.strip())
    word_limited = _limit_word_count(normalized, TITLE_MAX_WORDS)
    if len(word_limited) <= max_length:
        return word_limited
    return _truncate_to_last_complete_word(word_limited, max_length)
