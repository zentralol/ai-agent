"""Internal tool used by the model to submit structured place selections."""

from __future__ import annotations

from langchain_core.tools import tool

from app.schemas.recommendations import PlaceRecommendationSelection
from app.schemas.tools import ToolResponse, ToolStatus

SELECT_RECOMMENDED_PLACES_TOOL_NAME = "select_recommended_places"


@tool
def select_recommended_places(
    recommendations: list[PlaceRecommendationSelection],
) -> str:
    """Submit the places to show as recommendations, in display order.

    Call this only after a nearby-place or attraction lookup when the final
    answer recommends one or more returned candidates. Use candidate_id values
    exactly as returned by the lookup and include only places you recommend.
    The order of the recommendations list becomes the card order.
    """

    result = ToolResponse(
        status=ToolStatus.SUCCESS,
        summary=f"Selected {len(recommendations)} place recommendations.",
        data={
            "recommendations": [
                recommendation.model_dump(mode="json")
                for recommendation in recommendations
            ]
        },
    )
    return result.model_dump_json()
