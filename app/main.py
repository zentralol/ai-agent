from fastapi import FastAPI

from app.api.agent import router as agent_router

app = FastAPI(
    title="Zentra Agent API",
    version="0.1.0",
    description=(
        "Backend-only conversational AI agent for Zentra. "
        "Exposes a health check and a streaming chat endpoint."
    ),
)

app.include_router(agent_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "zentra-agent"}
