"""Tests for the multi-day planner orchestration."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

from app.agent.planner import MultiDayPlanner
from app.agent.trip_state import TripState
from app.schemas.events import (
    ToolFinishedEvent,
    ToolStartedEvent,
)
from app.schemas.tools import ToolResponse, ToolStatus
from app.tools.itinerary import PLAN_ITINERARY_TOOL_NAME


class _FakePlannerModel(BaseChatModel):
    """A model that never actually calls tools in planner tests."""

    @property
    def _llm_type(self) -> str:
        return "fake-planner"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=BaseMessage(content="", type="ai"))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=BaseMessage(content="", type="ai"))])


class _FakeItineraryTool(BaseTool):
    name: str = PLAN_ITINERARY_TOOL_NAME
    description: str = "Fake day planner"
    calls: list[dict[str, Any]] = []

    def __init__(self) -> None:
        super().__init__()
        self.calls = []

    async def _arun(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        date = kwargs.get("anchor_time", "")[:10]
        place_id = f"place-{date}"
        response = ToolResponse(
            status=ToolStatus.SUCCESS,
            summary=f"Itinerary built: 2 stops for {date}.",
            data={
                "stops": [
                    {
                        "time": "10:00",
                        "place_id": place_id,
                        "place_name": f"Stop on {date}",
                        "candidate_id": f"itinerary:{place_id}",
                        "lat": 40.7,
                        "lon": -73.9,
                        "neighborhood": "Manhattan",
                        "category": "park",
                        "crowd_category": "Quiet",
                        "hours": "Open",
                        "why_recommended": f"Best of {date}",
                    }
                ],
                "candidates": [
                    {
                        "candidate_id": f"itinerary:{place_id}",
                        "name": f"Stop on {date}",
                        "lat": 40.7,
                        "lng": -73.9,
                        "time": "10:00",
                        "neighborhood": "Manhattan",
                        "category": "park",
                        "crowd_category": "Quiet",
                        "hours": "Open",
                        "why_recommended": f"Best of {date}",
                    }
                ],
            },
        )
        return response.model_dump_json()

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError()


@pytest.fixture
def planner_tool() -> _FakeItineraryTool:
    return _FakeItineraryTool()


@pytest.fixture
def planner_model() -> _FakePlannerModel:
    return _FakePlannerModel()


async def test_plan_multi_day_spans_start_and_end_date(
    planner_model: _FakePlannerModel, planner_tool: _FakeItineraryTool
) -> None:
    planner = MultiDayPlanner()
    state = TripState(num_days=3, start_date="2026-07-12", anchor_place="Manhattan")

    events, data, updated_state = await planner.plan_multi_day(
        None, state, planner_model, (planner_tool,)
    )

    assert data.start_date == "2026-07-12"
    assert data.end_date == "2026-07-14"
    assert len(data.items) == 3
    assert updated_state.day_plans


async def test_plan_multi_day_calls_tool_once_per_day(
    planner_model: _FakePlannerModel, planner_tool: _FakeItineraryTool
) -> None:
    planner = MultiDayPlanner()
    state = TripState(num_days=2, start_date="2026-07-12", anchor_place="Manhattan")

    await planner.plan_multi_day(None, state, planner_model, (planner_tool,))

    assert len(planner_tool.calls) == 2
    assert planner_tool.calls[0]["anchor_time"] == "2026-07-12T10:00:00"
    assert planner_tool.calls[1]["anchor_time"] == "2026-07-13T10:00:00"


async def test_plan_multi_day_emits_tool_started_and_finished(
    planner_model: _FakePlannerModel, planner_tool: _FakeItineraryTool
) -> None:
    planner = MultiDayPlanner()
    state = TripState(num_days=2, start_date="2026-07-12", anchor_place="Manhattan")

    events, _, _ = await planner.plan_multi_day(None, state, planner_model, (planner_tool,))

    started = [e for e in events if isinstance(e, ToolStartedEvent)]
    finished = [e for e in events if isinstance(e, ToolFinishedEvent)]
    assert len(started) == 2
    assert len(finished) == 2
    assert all(e.tool_name == PLAN_ITINERARY_TOOL_NAME for e in started)


async def test_plan_multi_day_deduplicates_cross_day_place_ids(
    planner_model: _FakePlannerModel,
) -> None:
    tool = _FakeItineraryTool()
    planner = MultiDayPlanner()
    state = TripState(num_days=2, start_date="2026-07-12", anchor_place="Manhattan")

    _, data, _ = await planner.plan_multi_day(None, state, planner_model, (tool,))

    place_ids = [item.candidate_id for item in data.items]
    assert len(place_ids) == len(set(place_ids))


async def test_modify_day_replans_only_target_day(
    planner_model: _FakePlannerModel, planner_tool: _FakeItineraryTool
) -> None:
    planner = MultiDayPlanner()
    state = TripState(num_days=2, start_date="2026-07-12", anchor_place="Manhattan")
    _, _, state = await planner.plan_multi_day(None, state, planner_model, (planner_tool,))
    original_call_count = len(planner_tool.calls)

    events, data, updated_state = await planner.modify_day(
        None, state, planner_model, (planner_tool,), "2026-07-13"
    )

    assert len(planner_tool.calls) == original_call_count + 1
    assert planner_tool.calls[-1]["anchor_time"].startswith("2026-07-13")
    assert data.start_date == "2026-07-12"
    assert data.end_date == "2026-07-13"
    assert "2026-07-13" in updated_state.day_plans


async def test_plan_multi_day_sets_summary(
    planner_model: _FakePlannerModel, planner_tool: _FakeItineraryTool
) -> None:
    planner = MultiDayPlanner()
    state = TripState(num_days=2, start_date="2026-07-12", anchor_place="Manhattan")

    _, data, _ = await planner.plan_multi_day(None, state, planner_model, (planner_tool,))

    assert data.summary
    assert "Jul 12" in data.summary or "July 12" in data.summary
