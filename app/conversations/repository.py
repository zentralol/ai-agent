"""Supabase-backed conversation persistence owned by the agent.

Reads recent history to build model context and writes user/assistant turns.
The agent uses the service role key, which bypasses row-level security, so
every read and write is explicitly scoped by the authenticated ``user_id`` that
the Express gateway resolved. A raw ``conversation_id`` is never trusted alone.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.config import Settings, get_settings
from app.conversations.title import title_from_user_message
from app.db.supabase_client import create_supabase_client
from supabase import AsyncClient

logger = logging.getLogger("zentra_agent.conversations.repository")

# Fixed to match the product's Supabase migrations (see my-app
# supabase/migrations/20250626000000_conversations_messages.sql).
CONVERSATIONS_TABLE = "conversations"
MESSAGES_TABLE = "messages"


@dataclass(frozen=True)
class ChatTurn:
    """A single stored message reduced to what the model needs for context."""

    role: str
    content: str


class ConversationRepository:
    """Read history and append turns for a single conversation."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncClient | None = None

    @property
    def is_configured(self) -> bool:
        return bool(
            self._settings.supabase_url and self._settings.supabase_service_role_key
        )

    async def get_owned_conversation(
        self, conversation_id: str, user_id: str
    ) -> Mapping[str, Any] | None:
        """Return the conversation row iff it exists and belongs to the user."""

        client = await self._get_client()
        if client is None:
            return None

        response = await (
            client.table(CONVERSATIONS_TABLE)
            .select("*")
            .eq("id", conversation_id)
            .eq("user_id", user_id)
            .is_("deleted_at", "null")
            .maybe_single()
            .execute()
        )
        if response is None or response.data is None:
            return None
        return response.data if isinstance(response.data, Mapping) else None

    async def load_recent_messages(
        self, conversation_id: str, limit: int
    ) -> list[ChatTurn]:
        """Return up to ``limit`` most recent messages in chronological order."""

        client = await self._get_client()
        if client is None:
            return []

        response = await (
            client.table(MESSAGES_TABLE)
            .select("role, content")
            .eq("conversation_id", conversation_id)
            .is_("deleted_at", "null")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows = response.data if response and isinstance(response.data, list) else []
        turns = [
            ChatTurn(role=str(row["role"]), content=str(row["content"]))
            for row in rows
            if isinstance(row, Mapping) and row.get("role") and row.get("content")
        ]
        turns.reverse()  # DB returned newest-first; model wants oldest-first.
        return turns

    async def is_duplicate_last_user(
        self, conversation_id: str, content: str
    ) -> bool:
        """True when the newest stored message is an identical user message.

        Guards against a resubmitted turn writing the same user message twice
        without needing a client-provided message id.
        """

        client = await self._get_client()
        if client is None:
            return False

        response = await (
            client.table(MESSAGES_TABLE)
            .select("role, content")
            .eq("conversation_id", conversation_id)
            .is_("deleted_at", "null")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = response.data if response and isinstance(response.data, list) else []
        if not rows:
            return False
        last = rows[0]
        return (
            isinstance(last, Mapping)
            and last.get("role") == "user"
            and last.get("content") == content
        )

    async def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        ui_message_id: str,
        model: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        parts: list[dict[str, Any]] | None = None,
    ) -> None:
        """Insert one message row. A DB trigger bumps the conversation's updated_at."""

        client = await self._get_client()
        if client is None:
            return

        await (
            client.table(MESSAGES_TABLE)
            .insert(
                {
                    "conversation_id": conversation_id,
                    "role": role,
                    "content": content,
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "metadata": {"ui_message_id": ui_message_id},
                    "parts": parts or [],
                }
            )
            .execute()
        )

    async def ensure_title(self, conversation_id: str, first_user_text: str) -> None:
        """Set a derived title on the conversation when it has none."""

        client = await self._get_client()
        if client is None:
            return

        title = title_from_user_message(first_user_text)
        if not title:
            return

        await (
            client.table(CONVERSATIONS_TABLE)
            .update({"title": title})
            .eq("id", conversation_id)
            .is_("title", "null")
            .execute()
        )

    async def load_trip_state(
        self, conversation_id: str, user_id: str
    ) -> dict[str, Any] | None:
        """Return the agent-managed trip state for a conversation, if any.

        The query is scoped by ``user_id`` because the agent uses the service-role
        key, which bypasses row-level security.
        """

        client = await self._get_client()
        if client is None:
            return None

        response = await (
            client.table(CONVERSATIONS_TABLE)
            .select("trip_state")
            .eq("id", conversation_id)
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        if response is None or response.data is None:
            return None
        data = response.data if isinstance(response.data, Mapping) else {}
        trip_state = data.get("trip_state")
        return trip_state if isinstance(trip_state, dict) else None

    async def save_trip_state(
        self, conversation_id: str, user_id: str, state: dict[str, Any]
    ) -> None:
        """Persist the agent-managed trip state for a conversation.

        The update is scoped by ``user_id`` because the agent uses the service-role
        key, which bypasses row-level security.
        """

        client = await self._get_client()
        if client is None:
            return

        await (
            client.table(CONVERSATIONS_TABLE)
            .update({"trip_state": state})
            .eq("id", conversation_id)
            .eq("user_id", user_id)
            .execute()
        )

    async def _get_client(self) -> AsyncClient | None:
        if self._client is None:
            self._client = await create_supabase_client(self._settings)
        return self._client


@lru_cache(maxsize=1)
def get_conversation_repository() -> ConversationRepository:
    """Return the cached conversation repository."""

    return ConversationRepository(get_settings())
