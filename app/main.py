import logging

from fastapi import FastAPI

from app.api.agent import router as agent_router
from app.logging_format import (
    build_colored_formatter,
    configure_structlog,
    suppress_noisy_loggers,
)


def _configure_console_logging() -> None:
    formatter = build_colored_formatter()
    logger = logging.getLogger("zentra_agent")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        configure_structlog(formatter=formatter)
        suppress_noisy_loggers()
        return

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    configure_structlog(formatter=formatter)
    suppress_noisy_loggers()


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
