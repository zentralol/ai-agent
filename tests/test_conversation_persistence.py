"""Tests for agent-owned conversation persistence and memory.

``run_agent_stream`` is driven directly with a fake chat model and a fake
conversation repository injected via monkeypatch, so no network or database is
touched.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import Field

from app.agent import runner as runner_module
from app.agent.runner import run_agent_stream
from app.conversations.repository import ChatTurn
from app.schemas.chat import AgentStreamRequest, ClientType
from app.schemas.events import ErrorEvent, MessageDeltaEvent
from app.tools.catalog import AGENT_TOOLS


class _RecordingModel(FakeMessagesListChatModel):
    """Fake chat model that records the messages it was asked to complete."""

    messages_by_call: list[list[BaseMessage]] = Field(default_factory=list)

    def __init__(self, text: str) -> None:
        super().__init__(responses=[AIMessage(content=text)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> _RecordingModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.messages_by_call.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class _FakeRepo:
    def __init__(
        self,
        *,
        conversation: dict[str, Any] | None,
        history: list[ChatTurn] | None = None,
        configured: bool = True,
        duplicate: bool = False,
    ) -> None:
        self.is_configured = configured
        self._conversation = conversation
        self._history = history or []
        self._duplicate = duplicate
        self.appended: list[tuple[str, str]] = []
        self.titled: list[str] = []

    async def get_owned_conversation(
        self, conversation_id: str, user_id: str
    ) -> dict[str, Any] | None:
        return self._conversation

    async def load_recent_messages(
        self, conversation_id: str, limit: int
    ) -> list[ChatTurn]:
        return list(self._history)

    async def is_duplicate_last_user(self, conversation_id: str, content: str) -> bool:
        return self._duplicate

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
        self.appended.append((role, content))

    async def ensure_title(self, conversation_id: str, first_user_text: str) -> None:
        self.titled.append(first_user_text)


def _install_repo(monkeypatch: pytest.MonkeyPatch, repo: _FakeRepo) -> None:
    monkeypatch.setattr(
        runner_module.conversation_repository,
        "get_conversation_repository",
        lambda: repo,
    )


def _request() -> AgentStreamRequest:
    return AgentStreamRequest(
        user_id="u1",
        message="hi",
        client_type=ClientType.WEB,
        conversation_id="conv-1",
    )


async def _collect(model: Any, request: AgentStreamRequest) -> list[Any]:
    return [event async for event in run_agent_stream(request, model, AGENT_TOOLS)]


def _delta_text(events: list[Any]) -> str:
    return "".join(e.text for e in events if isinstance(e, MessageDeltaEvent))


@pytest.mark.asyncio
async def test_loads_history_as_context_and_writes_both_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = [
        ChatTurn(role="user", content="earlier question"),
        ChatTurn(role="assistant", content="earlier answer"),
        ChatTurn(role="user", content="hi"),
    ]
    repo = _FakeRepo(conversation={"title": None}, history=history)
    _install_repo(monkeypatch, repo)
    model = _RecordingModel("Answer")

    events = await _collect(model, _request())

    assert _delta_text(events) == "Answer"
    assert ("user", "hi") in repo.appended
    assert ("assistant", "Answer") in repo.appended
    assert repo.titled == ["hi"]

    seen = model.messages_by_call[0]
    contents = [m.content for m in seen]
    assert "earlier question" in contents
    assert "earlier answer" in contents


@pytest.mark.asyncio
async def test_rejects_conversation_not_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _FakeRepo(conversation=None)
    _install_repo(monkeypatch, repo)
    model = _RecordingModel("should not be used")

    events = await _collect(model, _request())

    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert events[0].code == "CONVERSATION_NOT_FOUND"
    assert repo.appended == []


@pytest.mark.asyncio
async def test_skips_duplicate_user_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _FakeRepo(
        conversation={"title": "Existing"},
        history=[ChatTurn(role="user", content="hi")],
        duplicate=True,
    )
    _install_repo(monkeypatch, repo)
    model = _RecordingModel("Answer")

    await _collect(model, _request())

    assert ("user", "hi") not in repo.appended
    assert ("assistant", "Answer") in repo.appended
    assert repo.titled == []  # title already set


@pytest.mark.asyncio
async def test_falls_back_to_single_turn_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _FakeRepo(conversation=None, configured=False)
    _install_repo(monkeypatch, repo)
    model = _RecordingModel("Solo")

    events = await _collect(model, _request())

    assert _delta_text(events) == "Solo"
    assert repo.appended == []
    contents = [m.content for m in model.messages_by_call[0]]
    assert "hi" in contents
