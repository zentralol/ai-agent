"""Model-callable tools exposed to the agent loop."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from app.tools.attractions import get_nearest_attractions
from app.tools.places import get_nearby_places
from app.tools.preferences import get_user_preferences

AGENT_TOOLS: tuple[BaseTool, ...] = (
    get_user_preferences,
    get_nearest_attractions,
    get_nearby_places,
)

