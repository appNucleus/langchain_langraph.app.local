# Runtime Architecture

This document describes the implementation present on the `release` baseline used for the naming-normalization change. Source and tests remain authoritative when this document and runtime behavior disagree.

## API lifecycle

1. `create_app()` resolves settings and constructs `ChatRuntimeAgent`.
2. FastAPI lifespan starts persistence, Ollama, and MCP according to required/optional policy.
3. `/api/chat` validates `ChatRequest`, normalizes identity, enters the same-conversation gate, and executes the graph.
4. `/api/chat/stream` wraps the same invocation in an SSE generator.
5. Lifespan shutdown closes Ollama, MCP, and persistence resources.

## Request lifecycle

```text
validate request
  -> normalize conversation/run identity
  -> acquire process-local conversation gate
  -> check process-local completed response registry
  -> inspect checkpoint for conflict or resume
  -> load conversation history and live inventory
  -> derive or preserve system instruction
  -> invoke checkpointed graph
  -> append history idempotently within the current store view
  -> cache completed response in the process-local registry
  -> return response and resume token
```

Conversation continuity, run identity, and LangGraph execution-thread identity are separate concepts. The gate and run registry are not distributed correctness boundaries.

## Graph adjacency

| Node | Possible next nodes |
|---|---|
| `plan` | `research`, `worker`, `terminate` |
| `research` | `worker`, `terminate` |
| `worker` | `verify`, `terminate` |
| `verify` | `advance`, `revise`, `research`, `replan`, `terminate` |
| `revise` | `verify`, `terminate` |
| `replan` | `worker`, `terminate` |
| `advance` | `research`, `worker`, `finalize` |
| `finalize` | `END`, `verify_final`, `terminate` |
| `verify_final` | `END`, `revise_final`, `terminate` |
| `revise_final` | `verify_final` |
| `terminate` | `END` |

## Model-call path

Model selection is role-based and constrained by live Ollama inventory. Planner, worker, verifier, reviser, synthesizer, and final verifier create structured agent calls using configured role models and bounded fallback behavior. The current execution budget records logical model-call increments; it is not yet an authoritative meter for every physical retry/fallback attempt.

## MCP-call path

The inventory service loads live MCP tools. The runtime router ranks compatible read-only tools and constructs arguments from each tool schema. `ToolExecutor` applies side-effect policy and budget checks before the MCP client call. Caller metadata does not authorize write actions.

## Evidence lifecycle

Retrieved tool results are normalized into typed evidence records with IDs, run/task/query metadata, content hashes, trust class, timestamps, truncation state, freshness, and quality fields. Current review must distinguish populated schema fields from values that are actually computed at runtime. Large evidence is not yet proven to use MinIO in the normal flow.

## Persistence lifecycle

Conversation history and checkpoints are configured independently. Memory is always available for local development. Redis or PostgreSQL can store history; memory or PostgreSQL can store checkpoints; MinIO can be enabled for artifacts. Optional startup failure falls back to memory with explicit degradation metadata. Outcome, checkpoint, and history commits are not yet one transactional unit.

## Streaming lifecycle

The current SSE implementation starts a background invocation, emits periodic generic work events, and cancels/awaits the task during generator cleanup. This is cancellation-aware polling, not graph-update or token streaming.

## Deployment lifecycle

```text
GitHub-hosted runner: checkout -> install -> compile -> Ruff -> deterministic pytest
self-hosted runner: checkout -> compose validation -> rollback point -> build -> deploy
post-deployment: smoke -> API checks -> backup -> cleanup or rollback -> logs
```

The image is rebuilt on the production runner and is not yet promoted by immutable digest from the test job.

## Compatibility register

| Compatibility item | Current behavior | Removal boundary |
|---|---|---|
| `thread_id` | Alias for `conversation_id` | Versioned API migration |
| `PHASE2_*` variables | Accepted as aliases for canonical `AGENT_*` settings | After deployment migration and deprecation window |
| `phase2_*` setting properties | Read-only aliases used by older internal callers | After all call sites use canonical settings |
| `phase5-v1` checkpoint namespace | Retained to read existing checkpoints | Explicit checkpoint/state migration only |
| legacy response metadata from base `ChatAgent` | Compatibility-only; production runtime emits `runtime_contract` | Versioned response contract |
