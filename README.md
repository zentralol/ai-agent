# Zentra Agent

Backend-only AI route planning agent for Zentra.

This repository will host the FastAPI agent service that sits behind the existing Express API gateway. Web and iOS clients should call the public Express `/api/v1` interfaces; this service should stay internal and expose route planning, itinerary generation, streaming agent responses, and MCP-backed tools for agent execution.

## Target Architecture

```text
Web / iOS
  -> Express backend gateway
    -> zentra-agent FastAPI
      -> LangGraph planning workflow
      -> FastMCP internal tools
      -> Existing Express prediction/recommendation interfaces
      -> Agent-owned Supabase tables for runs, traces, and plans
```

## Planned Stack

- FastAPI for HTTP and streaming endpoints.
- LangGraph as the main stateful agent orchestration runtime.
- LangChain components only where useful for models, prompts, and structured output.
- FastMCP for internal tool exposure.
- Pydantic for request, response, state, and tool schemas.
- httpx for internal calls to the existing Zentra Express backend.
- pytest, respx, ruff, and mypy for verification.

## Local Development

```bash
cd zentra-agent
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload --port 8010
```

Current endpoints are placeholders until the implementation phases begin:

- `GET /health`
- `POST /api/v1/itineraries`
- `POST /api/v1/routes/crowd-aware`
- `POST /api/v1/agent/stream`

See [DEVELOPMENT_PLAN.md](./DEVELOPMENT_PLAN.md) for the full implementation plan.

