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
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload --port 8010
```

Endpoints:

- `GET /health` — liveness check.
- `POST /api/v1/agent/stream` — streams typed chat events as Server-Sent Events.

### Configure the LLM

Set these in `.env` (OpenAI-compatible provider, e.g. DeepSeek on SenseNova):

```bash
LLM_MODEL=deepseek-v4-flash
LLM_API_KEY=sk-...
# Base URL must end at the API root; the client appends /chat/completions.
LLM_BASE_URL=https://token.sensenova.cn/v1
```

If `LLM_API_KEY` is unset, the endpoint returns a deterministic placeholder reply
instead of calling a model.

### Talk to the agent from the terminal

Start the service, then send a chat request. The response is an SSE stream: each
`message_delta` event carries a text chunk, and the stream ends with `done`.

```bash
# 1. Start the service (leave running in one terminal)
uv run uvicorn app.main:app --reload --port 8010

# 2. Health check
curl -s localhost:8010/health

# 3. Stream a chat response (-N disables curl buffering so tokens appear live)
curl -N -X POST localhost:8010/api/v1/agent/stream \
  -H 'content-type: application/json' \
  -d '{
        "user_id": "u1",
        "message": "Recommend one quiet travel spot in one sentence.",
        "client_type": "web"
      }'

# 4. With a conversation id and preferences (e.g. reply in Chinese)
curl -N -X POST localhost:8010/api/v1/agent/stream \
  -H 'content-type: application/json' \
  -d '{
        "user_id": "u1",
        "message": "用一句话推荐一个适合避开人群的旅行地点",
        "client_type": "web",
        "conversation_id": "c1",
        "preferences": { "language": "zh" }
      }'
```

To see only the streamed text (strip the SSE envelope):

```bash
curl -sN -X POST localhost:8010/api/v1/agent/stream \
  -H 'content-type: application/json' \
  -d '{"user_id":"u1","message":"Hello!","client_type":"web"}' \
  | sed -n 's/^data: //p' | jq -r 'select(.type=="message_delta") | .text' | tr -d '\n'; echo
```

See [DEVELOPMENT_PLAN.md](./DEVELOPMENT_PLAN.md) for the full implementation plan.
