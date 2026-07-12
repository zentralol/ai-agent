"""Tests for trip_state persistence scoped by user_id."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.conversations.repository import ConversationRepository


class _FakeClient:
    """Captures Supabase query chains."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self._response: dict[str, Any] | None = None

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(self, name)

    def set_response(self, data: dict[str, Any] | None) -> None:
        self._response = data


class _FakeQuery:
    def __init__(self, client: _FakeClient, table: str) -> None:
        self._client = client
        self._table = table
        self._chain: list[str] = [table]

    def select(self, column: str) -> _FakeQuery:
        self._chain.append(f"select:{column}")
        return self

    def update(self, values: dict[str, Any]) -> _FakeQuery:
        self._chain.append(f"update:{values}")
        return self

    def eq(self, column: str, value: Any) -> _FakeQuery:
        self._chain.append(f"eq:{column}={value}")
        return self

    def maybe_single(self) -> _FakeQuery:
        self._chain.append("maybe_single")
        return self

    def execute(self) -> Any:
        self._client.calls.append(self._chain)
        return asyncio.sleep(0, result=_FakeResponse(self._client._response or {}))


class _FakeResponse:
    def __init__(self, data: dict[str, Any] | None) -> None:
        self.data = data


class _RepoWithClient(ConversationRepository):
    def __init__(self, client: _FakeClient) -> None:  # type: ignore[override]
        self._client = client

    async def _get_client(self) -> _FakeClient:  # type: ignore[override]
        return self._client


@pytest.mark.asyncio
async def test_load_trip_state_scopes_by_user_id() -> None:
    client = _FakeClient()
    client.set_response({"trip_state": {"version": 1}})
    repo = _RepoWithClient(client)

    result = await repo.load_trip_state("conv-1", "user-1")

    assert result == {"version": 1}
    assert client.calls == [
        ["conversations", "select:trip_state", "eq:id=conv-1", "eq:user_id=user-1", "maybe_single"]
    ]


@pytest.mark.asyncio
async def test_save_trip_state_scopes_by_user_id() -> None:
    client = _FakeClient()
    repo = _RepoWithClient(client)
    state = {"version": 1, "mode": "multi"}

    await repo.save_trip_state("conv-1", "user-1", state)

    assert client.calls == [
        ["conversations", f"update:{{'trip_state': {state}}}", "eq:id=conv-1", "eq:user_id=user-1"]
    ]


@pytest.mark.asyncio
async def test_load_trip_state_returns_none_for_missing_state() -> None:
    client = _FakeClient()
    client.set_response({"trip_state": None})
    repo = _RepoWithClient(client)

    result = await repo.load_trip_state("conv-1", "user-1")

    assert result is None
