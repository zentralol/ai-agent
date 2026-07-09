import logging

from fastapi import FastAPI

from app.api.agent import router as agent_router


def _configure_console_logging() -> None:
    logger = logging.getLogger("zentra_agent")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)


_configure_console_logging()

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
