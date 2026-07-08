# Zentra Agent Development Plan

## 1. Objective

Build a backend-only conversational AI agent for Zentra.

The agent's job is to understand user messages, manage an AI conversation, decide when backend capabilities are needed, call those capabilities through tools, and stream grounded responses back to Web and iOS through the existing Express gateway.

This repository must stay focused on AI behavior. Deterministic product logic such as crowd-aware route computation, itinerary construction, route scoring, prediction fallback, recommendation ranking, and preference normalization belongs in the backend gateway and related backend modules.

## 2. Architectural Decision

Use one new FastAPI service named `zentra-agent`.

The service will contain:

- A single chat/streaming endpoint called by the existing Express gateway.
- A LangGraph workflow for AI conversation orchestration.
- Purpose-built FastMCP tools mounted inside the same FastAPI process.
- Tool adapters that call backend-owned capabilities through Express.
- Direct Supabase access only for AI-owned data such as `agent_runs`, `agent_tool_traces`, and optional conversation diagnostics.

Do not expose route planning or itinerary endpoints from this service. If the product later needs endpoints such as `/api/v1/routes/crowd-aware` or `/api/v1/itineraries`, those should be implemented in the backend repository and consumed by this agent as tools.

Do not start with a separate FastMCP deployment. Split FastMCP into its own service only after multiple agents or external systems need the same AI tool surface independently.

## 3. Target Topology

```text
Web / iOS chat UI
  -> Express backend gateway
    -> zentra-agent FastAPI
      -> LangGraph conversation workflow
      -> FastMCP tools mounted at /internal/mcp
      -> Express backend capabilities
      -> Supabase AI trace tables
```

The Express gateway remains the public backend entry point.

Responsibilities of Express:

- Verify Clerk authentication.
- Resolve `userId`.
- Load and normalize user preferences.
- Validate public request shape.
- Own product APIs such as prediction, recommendations, routes, and itineraries.
- Call `zentra-agent` with an internal token or short-lived internal JWT.
- Proxy streaming responses when required.
- Keep public `/api/v1` response format stable.

Responsibilities of `zentra-agent`:

- Run the AI conversation graph.
- Decide when backend capabilities are needed.
- Call backend-owned capabilities through narrow tools.
- Validate tool inputs and outputs.
- Generate responses grounded in tool results.
- Persist AI run metadata and tool traces.

## 4. Non-Goals

- Do not let Web or iOS call MCP directly.
- Do not implement crowd-aware route computation in this repository.
- Do not implement itinerary construction in this repository.
- Do not implement prediction fallback or recommendation ranking in this repository.
- Do not let the agent bypass Express prediction and route interfaces for the first release.
- Do not auto-convert every REST endpoint into MCP tools.
- Do not store client secrets in Web or iOS.

## 5. Planned Interfaces

### Express-Facing Endpoint

`POST /api/v1/agent/stream`

Streams AI chat responses and structured events for chat-like interactions. The gateway should proxy this endpoint to Web/iOS.

The endpoint should support stream events such as:

- `message_delta`
- `tool_started`
- `tool_finished`
- `backend_capability_result`
- `warning`
- `done`
- `error`

### Health Endpoint

`GET /health`

Used by deployment, monitoring, Docker, and CI checks.

### Internal MCP Endpoint

`/internal/mcp`

Mounted FastMCP endpoint for internal agent tool access. It should be protected by network controls and internal authentication. It is not a public client interface.

## 6. Initial Tool Set

Design every tool schema first. Tool names must be stable, explicit, and narrow.

Each tool response should include:

- `status`: `success`, `warning`, or `error`.
- `summary`: one-line result.
- `data`: typed payload.
- `next_actions`: actionable follow-ups.
- `artifacts`: IDs, paths, trace IDs, or generated plan IDs.

Initial MCP tools:

1. `predict_crowd_batch`
   - Calls the existing Express `/api/v1/predictions/batch`.
   - Returns normalized prediction objects plus warnings.

2. `get_crowd_forecast`
   - Calls the existing Express `/api/v1/predictions/forecast`.
   - Returns time-series crowd estimates for a coordinate.

3. `get_quieter_recommendations`
   - Calls the existing Express `/api/v1/recommendations`.
   - Returns nearby quieter H3 areas.

4. `get_crowd_aware_routes`
   - Calls a backend-owned route endpoint after that endpoint exists in the backend repository.
   - The agent must not compute or score routes itself.

5. `get_itinerary_plan`
   - Calls a backend-owned itinerary endpoint after that endpoint exists in the backend repository.
   - The agent must not implement deterministic itinerary construction itself.

6. `persist_agent_run`
   - Writes AI run metadata and tool trace references to Supabase AI-owned tables.

## 7. Agent Workflow

Use LangGraph for the main conversation workflow.

Initial graph:

1. `validate_request`
   - Validate the request schema and internal auth context.

2. `resolve_intent`
   - Classify whether the user needs general travel guidance, a crowd answer, route help, itinerary help, or a clarification.

3. `load_context`
   - Receive normalized user preferences from Express.
   - Load prior agent state only if a conversation ID is provided.

4. `maybe_clarify`
   - Ask a concise follow-up question if required facts are missing and cannot be safely defaulted.

5. `select_tools`
   - Choose backend capability tools based on the user intent.

6. `call_tools`
   - Call prediction, forecast, recommendation, backend route, or backend itinerary tools.

7. `synthesize_response`
   - Generate user-facing explanation grounded only in tool results and provided context.

8. `validate_output`
   - Validate final stream events and structured payloads against Pydantic schemas.

9. `persist_trace`
   - Persist run summary and tool traces.

## 8. Data Model Plan

Create AI-owned tables later through backend-managed migrations.

Suggested tables:

- `agent_runs`
  - `id`
  - `user_id`
  - `conversation_id`
  - `client_type`
  - `request_kind`
  - `status`
  - `model`
  - `started_at`
  - `finished_at`
  - `error_code`

- `agent_tool_traces`
  - `id`
  - `agent_run_id`
  - `tool_name`
  - `backend_endpoint`
  - `tool_input_hash`
  - `status`
  - `summary`
  - `latency_ms`
  - `created_at`

Do not create route plan or itinerary product tables in this repository. If needed, those tables should be owned by the backend product domain.

## 9. Development Phases

### Phase 0: Repository Baseline

Deliverables:

- FastAPI skeleton.
- English development plan.
- Package metadata.
- Empty module layout.
- Basic health endpoint.
- Placeholder chat stream endpoint.

Acceptance criteria:

- Repository exists as an independent git repository.
- `README.md` explains purpose and local setup.
- `DEVELOPMENT_PLAN.md` documents AI-only scope and phases.

### Phase 1: Chat Contracts and Schemas

Deliverables:

- Pydantic schemas for chat stream requests, stream events, tool responses, preferences snapshot, and internal auth context.
- OpenAPI docs for `/api/v1/agent/stream`.

Acceptance criteria:

- Request and response schemas can be validated without external services.
- Schema tests cover missing user context, malformed preferences, unsupported client type, and invalid stream event payloads.

### Phase 2: Express Gateway Integration

Deliverables:

- Internal auth middleware for Express-to-agent calls.
- Documented headers and internal token/JWT format.
- Express gateway integration plan for proxying `/api/v1/agent/stream`.

Acceptance criteria:

- Agent rejects unauthenticated direct calls.
- Agent accepts calls with valid internal auth.
- Express can pass `userId`, request ID, client type, conversation ID, and normalized preferences.

### Phase 3: MCP Tool Layer

Deliverables:

- FastMCP server mounted inside FastAPI at `/internal/mcp`.
- Tools that call backend-owned capabilities through Express.
- Structured error responses with retry guidance and stop conditions.

Acceptance criteria:

- MCP tools can be tested with in-memory transport.
- Tool schemas are narrow and purpose-built.
- No public Web/iOS client needs MCP knowledge.
- No tool implements route scoring or itinerary construction locally.

### Phase 4: LangGraph Conversation Skeleton

Deliverables:

- LangGraph state model.
- Graph nodes for validation, intent resolution, tool selection, tool execution, response synthesis, output validation, and persistence.
- Deterministic fallback response when the LLM is unavailable.

Acceptance criteria:

- A mock chat request completes end to end using fake backend tools.
- Graph state can be inspected in tests.
- Failed backend capability calls return recoverable agent states.

### Phase 5: Chat Agent MVP

Deliverables:

- Implement `/api/v1/agent/stream`.
- Stream text deltas and structured tool events.
- Support route-related and crowd-related questions by calling backend tools.

Acceptance criteria:

- Web can render incremental assistant text.
- Web and iOS can receive structured tool status events without parsing prose.
- Stream failures include a final recoverable error event.
- Route and itinerary answers are grounded in backend capability outputs.

### Phase 6: Backend Capability Expansion

Deliverables:

- Add tool adapters for backend-owned `/api/v1/routes/crowd-aware` after it exists in the backend.
- Add tool adapters for backend-owned `/api/v1/itineraries` after it exists in the backend.
- Add tests that verify the agent treats those endpoints as external capabilities.

Acceptance criteria:

- The agent can explain backend route or itinerary results.
- The agent does not recompute, rerank, or override backend-selected plans.
- Missing backend capability responses produce clear user-facing warnings.

### Phase 7: Persistence, Observability, and Evaluation

Deliverables:

- Agent run persistence.
- Tool trace persistence.
- Structured logs with request IDs.
- Evaluation fixtures for chat quality and grounding.
- Regression tests for tool selection behavior.

Acceptance criteria:

- Every agent run has a trace ID.
- Every tool call has status, latency, and summary.
- Test suite covers success, partial backend failure, and tool timeout cases.

### Phase 8: Production Hardening

Deliverables:

- Dockerfile.
- Health and readiness checks.
- Timeout and retry policy.
- Rate limit strategy coordinated with Express.
- Deployment environment documentation.
- Security review for internal tokens, Supabase credentials, and MCP exposure.

Acceptance criteria:

- MCP endpoint is not publicly accessible.
- Agent cannot be called without internal auth.
- LLM provider failures degrade gracefully.
- Backend gateway timeout is longer than agent tool timeout budget.

## 10. Testing Strategy

Use four levels of tests:

1. Schema tests
   - Validate chat request, stream event, and tool payloads.

2. Adapter tests
   - Mock Express backend with `respx`.
   - Verify retries, timeouts, error normalization, and warning handling.

3. Graph tests
   - Run LangGraph with fake backend tools.
   - Assert state transitions and stream events.

4. Evaluation tests
   - Curated chat scenarios.
   - Check that the agent calls tools for factual crowd and route questions.
   - Check that the agent asks clarifying questions when required route facts are missing.
   - Check that explanations cite available data only.

## 11. Risk Register

1. Express-to-agent-to-Express call loop
   - Mitigation: agent only calls backend capability endpoints, never public agent endpoints.

2. LLM hallucinated route facts
   - Mitigation: final answers must be grounded in backend tool outputs.

3. Tool surface too broad
   - Mitigation: use purpose-built tools instead of auto-converting the whole backend.

4. Product logic leaks into the AI repository
   - Mitigation: route computation, itinerary construction, and ranking stay in the backend repository.

5. MCP accidentally exposed publicly
   - Mitigation: mount under `/internal/mcp`, protect with network policy and internal auth.

6. Streaming complexity
   - Mitigation: define a small stream event schema before frontend integration.

## 12. First Implementation Backlog

1. Add Pydantic schemas for chat requests and stream events.
2. Add settings loader with environment validation.
3. Add internal auth middleware.
4. Add Express backend client with typed methods.
5. Add fake backend tool implementations for local graph tests.
6. Add LangGraph skeleton with a mock conversation flow.
7. Add FastMCP mounted app and first `predict_crowd_batch` tool.
8. Add `/api/v1/agent/stream` MVP.
9. Add tool trace persistence.
10. Add Web chat integration through Express gateway.

## 13. References

- FastMCP FastAPI integration: https://gofastmcp.com/integrations/fastapi
- MCP transports: https://modelcontextprotocol.io/specification/2025-11-25/basic/transports
- LangGraph overview: https://docs.langchain.com/oss/python/langgraph/overview
