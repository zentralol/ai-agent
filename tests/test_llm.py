"""Tests for the OpenAI-compatible chat model factory."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessageChunk
from pydantic import SecretStr

from app.config import get_settings
from app.llm import _ToolCallSafeChatOpenAI, _with_tool_call_indexes, get_chat_model


@pytest.fixture(autouse=True)
def _clear_model_caches() -> None:
    get_settings.cache_clear()
    get_chat_model.cache_clear()


def test_get_chat_model_streams_provider_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "secret-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")

    model = get_chat_model()

    assert model is not None
    assert model.streaming is True


def test_with_tool_call_indexes_backfills_missing_index() -> None:
    chunk = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {"name": "get_user_preferences", "arguments": ""},
                        }
                    ]
                }
            }
        ]
    }

    patched = _with_tool_call_indexes(chunk)

    assert patched["choices"][0]["delta"]["tool_calls"][0]["index"] == 0


def test_with_tool_call_indexes_backfills_continuation_chunk_without_index() -> None:
    chunk = {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": '{"a": 1}'}}]}}]}

    patched = _with_tool_call_indexes(chunk)

    assert patched["choices"][0]["delta"]["tool_calls"][0]["index"] == 0


def test_with_tool_call_indexes_leaves_well_formed_chunk_untouched() -> None:
    chunk = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"index": 0, "id": "call-1", "function": {"name": "a", "arguments": ""}},
                        {"index": 1, "id": "call-2", "function": {"name": "b", "arguments": ""}},
                    ]
                }
            }
        ]
    }

    patched = _with_tool_call_indexes(chunk)

    assert patched == chunk


def test_with_tool_call_indexes_ignores_chunks_without_tool_calls() -> None:
    chunk = {"choices": [{"delta": {"content": "hello"}}]}

    assert _with_tool_call_indexes(chunk) == chunk


def test_with_tool_call_indexes_ignores_chunks_without_choices() -> None:
    chunk: dict[str, object] = {"usage": {"total_tokens": 5}}

    assert _with_tool_call_indexes(chunk) == chunk


def test_streamed_tool_call_reconstructs_when_provider_omits_index() -> None:
    """Regression test for the SenseNova/DeepSeek missing-``index`` quirk.

    Without the fix, LangChain's raw ``rtc["index"]`` lookup raises ``KeyError``
    on each delta below, which is swallowed and drops the tool call entirely -
    the merged message would end up with no tool call at all.
    """

    model = _ToolCallSafeChatOpenAI(api_key=SecretStr("dummy"), model="dummy-model")
    raw_deltas = [
        {
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "get_user_preferences",
                                    "arguments": "",
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"function": {"arguments": '{"categories": '}},
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"function": {"arguments": '["crowd"]}'}},
                        ]
                    }
                }
            ]
        },
    ]

    generation_chunks = [
        model._convert_chunk_to_generation_chunk(delta, AIMessageChunk, {}) for delta in raw_deltas
    ]
    assert all(chunk is not None for chunk in generation_chunks)

    merged = generation_chunks[0].message
    for chunk in generation_chunks[1:]:
        merged = merged + chunk.message

    assert len(merged.tool_calls) == 1
    tool_call = merged.tool_calls[0]
    assert tool_call["name"] == "get_user_preferences"
    assert tool_call["id"] == "call-1"
    assert tool_call["args"] == {"categories": ["crowd"]}
