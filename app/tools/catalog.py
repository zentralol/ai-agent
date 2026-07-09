"""Model-callable tools exposed to the agent loop."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from app.tools.preferences import get_user_preferences

AGENT_TOOLS: tuple[BaseTool, ...] = (get_user_preferences,)

