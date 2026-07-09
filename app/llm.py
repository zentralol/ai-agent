"""LLM client factory.

Builds an OpenAI-compatible chat model (DeepSeek/SenseNova/etc.) from settings.
Returns ``None`` when no API key is configured so callers can fall back to a
deterministic response instead of failing.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.messages import BaseMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.config import get_settings

DEFAULT_TEMPERATURE = 0.7


class _ToolCallSafeChatOpenAI(ChatOpenAI):
    """``ChatOpenAI`` that tolerates providers omitting tool-call ``index``.

    SenseNova/DeepSeek-compatible endpoints only ever stream one tool call at a
    time and some of them omit the ``index`` field on the delta entirely.
    LangChain's delta parser does ``raw_tool_call["index"]`` and, on
    ``KeyError``, silently drops *every* tool-call chunk in that delta -
    breaking tool calling under streaming. Since these providers never
    interleave parallel tool calls, defaulting a missing index to ``0`` is
    safe and lets LangChain's chunk-merging reconstruct the call correctly.
    """

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict[str, Any],
        default_chunk_class: type[BaseMessageChunk],
        base_generation_info: dict[str, Any] | None,
    ) -> ChatGenerationChunk | None:
        return super()._convert_chunk_to_generation_chunk(
            _with_tool_call_indexes(chunk),
            default_chunk_class,
            base_generation_info,
        )


def _with_tool_call_indexes(chunk: dict[str, Any]) -> dict[str, Any]:
    """Backfill a missing ``index`` on a streamed tool-call delta."""

    choices = chunk.get("choices")
    if not choices:
        return chunk

    delta = choices[0].get("delta")
    tool_calls = delta.get("tool_calls") if delta else None
    if not tool_calls or all("index" in call for call in tool_calls):
        return chunk

    patched_delta = {
        **delta,
        "tool_calls": [call if "index" in call else {**call, "index": 0} for call in tool_calls],
    }
    return {
        **chunk,
        "choices": [{**choices[0], "delta": patched_delta}, *choices[1:]],
    }


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI | None:
    """Return a cached streaming chat model, or ``None`` if unconfigured."""

    settings = get_settings()
    if not settings.llm_api_key:
        return None

    return _ToolCallSafeChatOpenAI(
        model=settings.llm_model,
        api_key=SecretStr(settings.llm_api_key),
        base_url=settings.llm_base_url,
        streaming=True,
        temperature=DEFAULT_TEMPERATURE,
    )
