"""Tests for intent classification and guardrails."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent.intent import IntentClassification, classify_intent
from app.agent.trip_state import ClarificationState, TripState


class _StubModel(BaseChatModel):
    """A model that returns a fixed structured output."""

    response: IntentClassification | None = None
    invocations: list[list[BaseMessage]] = []

    def __init__(self, response: IntentClassification | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.response = response or IntentClassification(intent="general")
        self.invocations = []

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.invocations.append(messages)
        generation = ChatGeneration(message=self.response)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.invocations.append(messages)
        generation = ChatGeneration(message=self.response)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "stub"

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _StubModel:
        self._structured_schema = schema
        return self

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        self.invocations.append(input if isinstance(input, list) else [input])
        return self.response


async def test_classify_intent_returns_model_output() -> None:
    expected = IntentClassification(
        intent="multi_day",
        num_days=3,
        start_date="2026-07-12",
        anchor_place="Manhattan",
        additional_context="family friendly",
    )
    model = _StubModel(response=expected)

    result = await classify_intent(model, "Plan 3 days in Manhattan", TripState(), "2026-07-11")

    assert result == expected


async def test_classify_intent_extracts_num_days_from_message() -> None:
    model = _StubModel(response=IntentClassification(intent="multi_day"))

    result = await classify_intent(model, "Plan a 5 day trip", TripState(), "2026-07-11")

    assert result.num_days == 5


async def test_classify_intent_extracts_date_range_from_message() -> None:
    model = _StubModel(response=IntentClassification(intent="multi_day", num_days=3))

    result = await classify_intent(
        model, "July 12-14 in Manhattan", TripState(), "2026-07-11"
    )

    assert result.start_date == "2026-07-12"


async def test_classify_intent_extracts_modify_target_day() -> None:
    model = _StubModel(response=IntentClassification(intent="modify_day"))

    result = await classify_intent(model, "Change day 2 to food tour", TripState(), "2026-07-11")

    assert result.modify_target == "day 2"


async def test_classify_intent_uses_today_for_relative_start() -> None:
    model = _StubModel(response=IntentClassification(intent="multi_day", num_days=2))

    result = await classify_intent(model, "Plan a trip starting today", TripState(), "2026-07-11")

    assert result.start_date == "2026-07-11"


async def test_classify_intent_uses_tomorrow_for_tomorrow() -> None:
    model = _StubModel(response=IntentClassification(intent="multi_day", num_days=2))

    result = await classify_intent(
        model, "Plan a trip starting tomorrow", TripState(), "2026-07-11"
    )

    assert result.start_date == "2026-07-12"


async def test_classify_intent_caps_at_two_clarification_rounds() -> None:
    model = _StubModel(
        response=IntentClassification(
            intent="clarify", missing_fields=["start_date"], question_to_user="When?"
        )
    )
    state = TripState(
        clarification=ClarificationState(missing=["start_date"], count=0)
    )
    for count in range(1, 4):
        result = await classify_intent(model, "I don't know", state, "2026-07-11")
        if count <= 2:
            assert result.intent == "clarify"
            state = state.model_copy(
                update={
                    "clarification": ClarificationState(
                        missing=["start_date"], count=count
                    )
                }
            )
        else:
            assert result.intent == "multi_day"
            assert result.start_date == "2026-07-12"
            assert result.num_days == 3


async def test_classify_intent_system_prompt_is_english() -> None:
    model = _StubModel(response=IntentClassification(intent="general"))

    await classify_intent(model, "I want to plan a trip", TripState(), "2026-07-11")

    assert model.invocations
    prompt_text = " ".join(str(m.content) for m in model.invocations[0])
    assert "intent" in prompt_text.lower()
    # Basic English check: the prompt and user message are ASCII-only.
    assert prompt_text.isascii()
