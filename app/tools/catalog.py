"""Model-callable tools exposed to the agent loop."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from app.tools.attractions import get_nearest_attractions
from app.tools.crowd import predict_crowd_level
from app.tools.itinerary import plan_itinerary
from app.tools.places import get_nearby_places
from app.tools.preferences import get_user_preferences
from app.tools.recommendations import select_recommended_places
from app.tools.recommendations_itinerary import get_place_recommendations

AGENT_TOOLS: tuple[BaseTool, ...] = (
    get_user_preferences,
    get_nearest_attractions,
    get_nearby_places,
    predict_crowd_level,
    select_recommended_places,
    plan_itinerary,
    get_place_recommendations,
)
