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
      -> Supabase user preferences + agent-owned run/trace tables
```

## Planned Stack

- FastAPI for health and streaming chat endpoints.
- LangGraph as the main stateful agent orchestration runtime.
- LangChain `create_agent` for model/tool orchestration.
- FastMCP for internal tool exposure.
- Pydantic for request, response, state, and tool schemas.
- httpx for internal calls to the existing Zentra Express backend.
- pytest, respx, ruff, and mypy for verification.

## Project Structure

```text
zentra-agent/
├── app/                     # Application package
│   ├── main.py              # FastAPI app: /health + mounts the agent router
│   ├── config.py            # Env-driven, frozen settings (pydantic-settings) + get_settings()
│   ├── llm.py               # OpenAI-compatible chat model factory (get_chat_model)
│   ├── agent/
│   │   ├── runner.py        # Runs one create_agent turn and emits StreamEvent objects
│   │   └── stream_adapter.py # LangChain v3 stream events → Zentra StreamEvent
│   ├── api/
│   │   └── agent.py         # POST /api/v1/agent/stream — SSE streaming endpoint
│   ├── tools/
│   │   ├── catalog.py       # AGENT_TOOLS list bound to the model
│   │   └── preferences.py   # @tool get_user_preferences backed by Supabase
│   └── schemas/             # Pydantic contracts (immutable, one concern per file)
│       ├── chat.py          # AgentStreamRequest, ClientType
│       ├── events.py        # Stream events as a `type`-discriminated union (StreamEvent)
│       ├── preferences.py   # PreferenceCategory and sanitized UserPreferences
│       └── tools.py         # ToolResponse envelope + ToolStatus
├── tests/                   # pytest suite (network-free)
│   ├── test_config.py       # Settings loading, defaults, immutability, caching
│   ├── test_schemas.py      # Request/event validation (Phase 1 acceptance cases)
│   ├── test_preferences_tool.py # Preference tool schema and Supabase fallback behavior
│   └── test_agent_stream.py     # Endpoint: LLM / fallback / error / 422 paths
├── docs/NOTES.md            # Scope notes
├── DEVELOPMENT_PLAN.md      # Full phased implementation plan
├── pyproject.toml           # Dependencies, ruff/mypy/pytest config
├── .env.example             # Environment variable template
└── README.md
```

Layering: `main` wires the app → `api/agent` handles HTTP/SSE → `agent/runner` runs a
LangChain `create_agent` graph → `agent/stream_adapter` translates LangChain events into
Zentra stream events → `tools/catalog` provides the LangChain tool list → `tools` owns
server-side capabilities → `schemas` define contracts → `config`/`llm` provide
configuration and the model client. Packages for later phases (`adapters` for backend
clients) will be added when those features land.

### Agent Runner

`agent/runner.py` owns one agent turn. It builds a LangChain agent with:

- `model`: the configured chat model from `llm.py`;
- `tools`: the server-owned tools from `tools/catalog.py`, converted to a list for
  LangChain;
- `system_prompt`: Zentra's travel-assistant behavior and tool-use rules.

The `cast(Any, create_agent(...))` call is only for static typing. LangChain returns a
compiled graph with a broad runtime surface, and mypy does not reliably infer methods
such as `astream_events`. The cast does not change runtime behavior; it keeps the
runner focused on orchestration while `agent/stream_adapter.py` handles event-shape
translation.

## Preference Lookup

Clients and the Express gateway should not send arbitrary preference snapshots in the
chat request. The model receives a narrow `get_user_preferences` tool schema and decides
whether stored preferences are needed for the current answer.

The agent uses LangChain `create_agent`: every model turn can return text, tool calls,
or both. LangChain v3 event-stream payloads stay behind the service boundary and are
adapted into Zentra's stable SSE events, including `tool_started` and `tool_finished`.
The graph continues until the model stops requesting tools or hits the bounded step
limit.

The tool:

- uses the authenticated `user_id` from the internal request context;
- lets the model request only narrow categories such as `crowd`, `transport`,
  `budget`, `accessibility`, `language`, and `interests`;
- queries Supabase with server-side credentials;
- returns a compact `ToolResponse` payload;
- treats stored preference text as data, not system instructions.

If Supabase is not configured, chat still works. The preference tool returns a warning and
the agent continues with neutral defaults or asks a clarifying question.

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

### Stream Contract

`/api/v1/agent/stream` exposes Zentra events, not raw LangChain events. Every SSE
`data:` frame contains a JSON object with `type` and a monotonic `sequence` number.

Public event types:

- `message_delta`: assistant text to append. With a streaming provider this is a
  token/text delta; test models or non-streaming providers may emit a full message
  in one chunk.
- `tool_started`: a tool invocation began. Includes `tool_name` and, when available,
  `tool_call_id`.
- `tool_finished`: a tool invocation completed. Includes `tool_name`, optional
  `tool_call_id`, and a normalized `ToolResponse`.
- `warning`: non-fatal service warning.
- `done`: successful terminal event. Includes `conversation_id` and optional usage
  metadata if the provider reports it.
- `error`: terminal failure event with stable `code` and human-readable `message`.

`metadata` is reserved for optional diagnostics or UI hints. Clients should switch on
`type` and ignore unknown optional fields.

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

### Local Testing

Start the service, then send a chat request. The response is an SSE stream:
`message_delta` carries assistant text chunks, and the stream ends with `done`.

1. Start the service (leave running in one terminal):

```bash
uv run uvicorn app.main:app --reload --port 8010
```

2. Health check:

```bash
curl -s localhost:8010/health
```

3. Stream a chat response (`-N` disables curl buffering so SSE events appear live):

```bash
curl -N -X POST localhost:8010/api/v1/agent/stream \
  -H 'content-type: application/json' \
  -d '{
        "user_id": "u1",
        "message": "Recommend one quiet travel spot in one sentence.",
        "client_type": "web"
      }'
```

4. With a conversation id. The model can call `get_user_preferences` when the answer
   needs stored preference context and Supabase is configured:

```bash
curl -N -X POST localhost:8010/api/v1/agent/stream \
  -H 'content-type: application/json' \
  -d '{
        "user_id": "u1",
        "message": "Recommend one quiet travel spot in one sentence.",
        "client_type": "web",
        "conversation_id": "c1"
      }'
```

5. See only the streamed text (strip the SSE envelope):

```bash
curl -sN -X POST localhost:8010/api/v1/agent/stream \
  -H 'content-type: application/json' \
  -d '{"user_id":"u1","message":"Hello!","client_type":"web"}' \
  | sed -n 's/^data: //p' | jq -r 'select(.type=="message_delta") | .text' | tr -d '\n'; echo
```

See [DEVELOPMENT_PLAN.md](./DEVELOPMENT_PLAN.md) for the full implementation plan.
