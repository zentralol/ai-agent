"""Registry for model-callable tools.

The registry is the only place that maps tool schemas to server-side executors.
Model-provided arguments are validated and combined with trusted request context
before a tool touches external systems.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from app.schemas.tools import ToolResponse, ToolStatus
from app.tools.preferences import (
    GET_USER_PREFERENCES_TOOL_NAME,
    GET_USER_PREFERENCES_TOOL_SCHEMA,
    UserPreferenceTool,
    parse_preference_categories,
)

ToolArgs = Mapping[str, object]
ToolSchema = dict[str, Any]
ToolExecutor = Callable[["ToolContext", ToolArgs], Awaitable[ToolResponse]]


@dataclass(frozen=True)
class ToolContext:
    """Trusted context injected by the agent service, never by the model."""

    user_id: str
    request_id: str | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class AgentTool:
    """A model-callable tool and its server-side executor."""

    name: str
    schema: ToolSchema
    execute: ToolExecutor


ToolRegistry = dict[str, AgentTool]


def build_tool_registry(preference_tool: UserPreferenceTool) -> ToolRegistry:
    """Build the tool registry for the current request dependencies."""

    async def execute_get_user_preferences(
        context: ToolContext, args: ToolArgs
    ) -> ToolResponse:
        categories = parse_preference_categories(args.get("categories"))
        if not categories:
            return ToolResponse(
                status=ToolStatus.WARNING,
                summary="The model requested user preferences without valid categories.",
                next_actions=[
                    "Continue without stored preferences for this response.",
                ],
            )

        return await preference_tool.get_user_preferences(
            user_id=context.user_id,
            categories=categories,
        )

    return {
        GET_USER_PREFERENCES_TOOL_NAME: AgentTool(
            name=GET_USER_PREFERENCES_TOOL_NAME,
            schema=GET_USER_PREFERENCES_TOOL_SCHEMA,
            execute=execute_get_user_preferences,
        )
    }


def tool_schemas(registry: ToolRegistry) -> list[ToolSchema]:
    """Return the model-facing schemas for all registered tools."""

    return [tool.schema for tool in registry.values()]

