# Zentra Agent Development Plan

## 1. Objective

Build a backend-only AI route planning agent that gives Zentra users personalized, crowd-aware route and itinerary recommendations across Web and iOS.

The agent must not live inside the frontend or mobile clients. Clients should call stable product interfaces through the existing Express gateway. The agent service should own planning, tool orchestration, structured outputs, tool traces, and agent-specific persistence.

## 2. Architectural Decision

Use one new FastAPI service named `zentra-agent`.

The service will contain:

- Public-internal FastAPI endpoints called by the existing Express gateway.
- A LangGraph workflow for route planning and itinerary generation.
- Purpose-built FastMCP tools mounted inside the same FastAPI process.
- Adapters for the existing Express prediction and recommendation interfaces.
- Direct Supabase access only for agent-owned tables such as `agent_runs`, `agent_tool_traces`, `route_plans`, and `itineraries`.

Do not start with a separate FastMCP deployment. Split FastMCP into its own service only after multiple agents or external systems need the same tools independently.

## 3. Target Topology

```text
Web / iOS
  -> Express backend gateway
    -> zentra-agent FastAPI
      -> LangGraph workflow
      -> FastMCP tools mounted at /internal/mcp
      -> Express prediction/recommendation interfaces
      -> Google Routes adapter, if moved from Web
      -> Supabase agent tables
```

The Express gateway remains the public backend entry point.

Responsibilities of Express:

- Verify Clerk authentication.
- Resolve `userId`.
- Load and normalize user preferences.
- Validate public request shape.
- Call `zentra-agent` with an internal token or short-lived internal JWT.
- Proxy streaming responses when required.
- Keep public `/api/v1` response format stable.

Responsibilities of `zentra-agent`:

- Run the planning graph.
- Decide when to call tools.
- Validate all tool inputs and outputs.
- Score and rank route options deterministically.
- Generate grounded explanations from tool results.
- Persist agent runs, tool traces, and generated plans.

## 4. Non-Goals

- Do not let Web or iOS call MCP directly.
- Do not let the agent bypass Express prediction interfaces for the first release.
- Do not use the LLM as the only route planner.
- Do not auto-convert every REST endpoint into MCP tools.
- Do not store client secrets in Web or iOS.

## 5. Planned Interfaces

### Express-Facing Endpoints

`POST /api/v1/itineraries`

Creates a personalized itinerary with ordered stops, arrival/departure windows, route summary, crowd predictions, reasons, and warnings.

`POST /api/v1/routes/crowd-aware`

Creates route options between an origin and destination. Each route includes duration, distance, segment-level crowd scores, route-level crowd score, tradeoffs, warnings, and a grounded reason.

`POST /api/v1/agent/stream`

Streams agent messages and structured plan updates for chat-like interactions. The gateway should proxy this endpoint to Web/iOS.

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

4. `compute_routes`
   - Uses the Google Routes adapter after the Web route computation is moved backend-side.
   - Returns route geometry, duration, distance, and mode.

5. `score_route_options`
   - Deterministic Python scoring tool.
   - Combines crowd score, duration, walking distance, preference fit, accessibility constraints, and warnings.

6. `persist_agent_run`
   - Writes run metadata and tool trace references to Supabase agent-owned tables.

7. `persist_route_plan`
   - Writes generated route plans and itineraries to Supabase agent-owned tables.

## 7. Agent Workflow

Use LangGraph for the main planning workflow.

Initial graph:

1. `validate_request`
   - Validate the request schema and internal auth context.

2. `resolve_intent`
   - Classify whether the user needs a direct route, an itinerary, a quieter alternative, or a clarification.

3. `load_context`
   - Receive normalized user preferences from Express.
   - Load prior agent state only if a conversation or plan ID is provided.

4. `maybe_clarify`
   - Ask a concise follow-up question if required fields are missing and cannot be safely defaulted.

5. `generate_candidates`
   - Produce route or itinerary candidates using deterministic rules and tool calls.

6. `call_tools`
   - Call route, prediction, forecast, and recommendation tools.

7. `score_candidates`
   - Rank candidates deterministically.

8. `synthesize_response`
   - Generate user-facing explanation grounded only in tool results.

9. `validate_output`
   - Validate final JSON against Pydantic schemas.

10. `persist_trace`
   - Persist run summary, selected plan, and tool traces.

## 8. Data Model Plan

Create agent-owned tables later through backend-managed migrations.

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
  - `tool_input_hash`
  - `status`
  - `summary`
  - `latency_ms`
  - `created_at`

- `route_plans`
  - `id`
  - `agent_run_id`
  - `user_id`
  - `origin`
  - `destination`
  - `target_time`
  - `selected_route`
  - `alternatives`
  - `warnings`
  - `created_at`

- `itineraries`
  - `id`
  - `agent_run_id`
  - `user_id`
  - `start_time`
  - `end_time`
  - `stops`
  - `route_summary`
  - `warnings`
  - `created_at`

## 9. Development Phases

### Phase 0: Repository Baseline

Deliverables:

- FastAPI skeleton.
- English development plan.
- Package metadata.
- Empty module layout.
- Basic health endpoint.

Acceptance criteria:

- Repository exists as an independent git repository.
- `README.md` explains purpose and local setup.
- `DEVELOPMENT_PLAN.md` documents architecture and phases.

### Phase 1: Contracts and Schemas

Deliverables:

- Pydantic schemas for route planning requests, itinerary requests, route options, stops, warnings, preferences, tool responses, and stream events.
- Internal auth context schema.
- OpenAPI docs for placeholder endpoints.

Acceptance criteria:

- Request and response schemas can be validated without external services.
- Schema tests cover invalid coordinates, missing target time, duplicate locations, unsupported modes, and malformed preferences.

### Phase 2: Express Gateway Integration

Deliverables:

- Internal client for Express gateway.
- Internal auth middleware for Express-to-agent calls.
- Documented headers and internal token/JWT format.
- Placeholder Express routes planned for `/api/v1/itineraries`, `/api/v1/routes/crowd-aware`, and `/api/v1/agent/stream`.

Acceptance criteria:

- Agent rejects unauthenticated direct calls.
- Agent accepts calls with valid internal auth.
- Express can pass `userId`, request ID, client type, and normalized preferences.

### Phase 3: MCP Tool Layer

Deliverables:

- FastMCP server mounted inside FastAPI at `/internal/mcp`.
- Tools for prediction batch, forecast, quieter recommendations, route computation, scoring, and persistence.
- Structured error responses with retry guidance and stop conditions.

Acceptance criteria:

- MCP tools can be tested with in-memory transport.
- Tool schemas are narrow and purpose-built.
- No public Web/iOS client needs MCP knowledge.

### Phase 4: LangGraph Planning Skeleton

Deliverables:

- LangGraph state model.
- Graph nodes for validation, intent resolution, tool execution, scoring, output validation, and persistence.
- Deterministic fallback path when the LLM is unavailable.

Acceptance criteria:

- A mock route planning request completes end to end using fake tools.
- Graph state can be inspected in tests.
- Failed tool calls return recoverable agent states.

### Phase 5: Crowd-Aware Route MVP

Deliverables:

- Implement `/api/v1/routes/crowd-aware`.
- Move or reimplement Google Routes computation as a backend adapter.
- Predict crowd for route-relevant points or segments.
- Score route alternatives using deterministic weights.

Acceptance criteria:

- Given origin, destination, target time, and preferences, the service returns ranked route options.
- Each route has a reason grounded in route and prediction data.
- The selected route changes when crowd tolerance changes.

### Phase 6: Itinerary MVP

Deliverables:

- Implement `/api/v1/itineraries`.
- Accept candidate POIs from the gateway or a future POI provider.
- Sequence stops with time windows.
- Predict crowd at each stop and score preference fit.

Acceptance criteria:

- The service returns a valid itinerary with ordered stops, predictions, route summary, reasons, and warnings.
- It handles missing POI metadata with explicit warnings.
- It never invents live hours or accessibility facts.

### Phase 7: Streaming Agent Interface

Deliverables:

- Implement `/api/v1/agent/stream`.
- Stream text deltas and structured plan events.
- Include event types such as `message_delta`, `tool_started`, `tool_finished`, `plan_patch`, `warning`, and `done`.

Acceptance criteria:

- Web can render incremental assistant text.
- Web and iOS can receive structured plan updates without parsing prose.
- Stream failures include a final recoverable error event.

### Phase 8: Persistence, Observability, and Evaluation

Deliverables:

- Agent run persistence.
- Tool trace persistence.
- Structured logs with request IDs.
- Evaluation fixtures for route scoring and itinerary quality.
- Regression tests for preference-sensitive decisions.

Acceptance criteria:

- Every agent run has a trace ID.
- Every tool call has status, latency, and summary.
- Test suite covers success, partial failure, and tool timeout cases.

### Phase 9: Production Hardening

Deliverables:

- Dockerfile.
- Health and readiness checks.
- Timeout and retry policy.
- Rate limit strategy.
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
   - Validate request, response, and tool payloads.

2. Adapter tests
   - Mock Express backend with `respx`.
   - Verify retries, timeouts, error normalization, and warning handling.

3. Graph tests
   - Run LangGraph with fake tools.
   - Assert state transitions and final structured output.

4. Evaluation tests
   - Curated route planning scenarios.
   - Check that lower crowd tolerance increases crowd avoidance.
   - Check that mobility constraints affect route scoring.
   - Check that explanations cite available data only.

## 11. Risk Register

1. Express-to-agent-to-Express call loop
   - Mitigation: agent only calls prediction/recommendation interfaces, never public agent endpoints.

2. LLM hallucinated route facts
   - Mitigation: final output must be schema-validated and grounded in tool outputs.

3. Tool surface too broad
   - Mitigation: use purpose-built tools instead of auto-converting the whole backend.

4. Preference schema drift
   - Mitigation: normalize preferences in Express and add contract tests.

5. MCP accidentally exposed publicly
   - Mitigation: mount under `/internal/mcp`, protect with network policy and internal auth.

6. Streaming complexity
   - Mitigation: define a small stream event schema before frontend integration.

## 12. First Implementation Backlog

1. Add Pydantic schemas for route planning and tool responses.
2. Add settings loader with environment validation.
3. Add internal auth middleware.
4. Add Express backend client with typed methods.
5. Add fake tool implementations for local graph tests.
6. Add LangGraph skeleton with deterministic mock flow.
7. Add FastMCP mounted app and first `predict_crowd_batch` tool.
8. Add `/api/v1/routes/crowd-aware` MVP.
9. Add tool trace persistence.
10. Add Web integration through Express gateway.

## 13. References

- FastMCP FastAPI integration: https://gofastmcp.com/integrations/fastapi
- MCP transports: https://modelcontextprotocol.io/specification/2025-11-25/basic/transports
- LangGraph overview: https://docs.langchain.com/oss/python/langgraph/overview

