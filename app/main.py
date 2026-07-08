from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

app = FastAPI(
    title="Zentra Agent API",
    version="0.1.0",
    default_response_class=ORJSONResponse,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "zentra-agent"}


@app.post("/api/v1/itineraries", status_code=501)
async def create_itinerary_placeholder() -> dict[str, object]:
    return {
        "success": False,
        "error": {
            "code": "NOT_IMPLEMENTED",
            "message": "Itinerary planning is scheduled for a later development phase.",
        },
    }


@app.post("/api/v1/routes/crowd-aware", status_code=501)
async def crowd_aware_route_placeholder() -> dict[str, object]:
    return {
        "success": False,
        "error": {
            "code": "NOT_IMPLEMENTED",
            "message": "Crowd-aware route planning is scheduled for a later development phase.",
        },
    }


@app.post("/api/v1/agent/stream", status_code=501)
async def agent_stream_placeholder() -> dict[str, object]:
    return {
        "success": False,
        "error": {
            "code": "NOT_IMPLEMENTED",
            "message": "Agent streaming is scheduled for a later development phase.",
        },
    }

