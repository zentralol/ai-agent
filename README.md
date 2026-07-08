# Zentra Agent

Backend-only conversational AI agent for Zentra.

This repository hosts the FastAPI agent service that sits behind the existing Express API gateway. Web and iOS clients should call the public Express `/api/v1` interfaces; this service should stay internal and expose only the AI chat/streaming interface plus MCP-backed tools for agent execution.

Deterministic product logic such as crowd-aware route computation, itinerary construction, route scoring, prediction fallback, and recommendation ranking belongs in the backend gateway and related backend modules, not in this repository. The AI agent may call those backend capabilities as tools, but it should not own their implementation.

## Target Architecture

```text
Web / iOS
  -> Express backend gateway
    -> zentra-agent FastAPI
      -> LangGraph conversation workflow
      -> FastMCP internal tools
      -> Existing Express backend capabilities
      -> Agent-owned Supabase tables for runs and traces
```

## Planned Stack

- FastAPI for health and streaming chat endpoints.
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
- `POST /api/v1/agent/stream`

See [DEVELOPMENT_PLAN.md](./DEVELOPMENT_PLAN.md) for the full implementation plan.
