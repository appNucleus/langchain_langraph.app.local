# Runtime Architecture

This document describes the runtime identity and durable-execution contract. Source and tests remain authoritative when documentation and runtime behavior disagree.

## API lifecycle

1. `create_app()` resolves settings and constructs `ChatRuntimeAgent`.
2. FastAPI lifespan starts conversation storage, the run repository, checkpoints, artifacts, Ollama, and MCP according to required/optional policy.
3. `/api/chat` validates `ChatRequest`, normalizes identity, resolves the durable run, acquires a conversation lease, and executes the graph.
4. `/api/chat/stream` wraps the same invocation in an SSE generator.
5. Lifespan shutdown closes all external resources.

## Request and run lifecycle

```text
validate request
  -> normalize conversation/run identity
  -> canonicalize and version request hash
  -> enter process-local fast-path gate
  -> create or resolve durable run record
  -> replay a terminal stored response when available
  -> validate explicit resume token and resumable status
  -> inspect checkpoint for conflict or resume
  -> acquire distributed conversation lease and fencing token
  -> renew lease while graph execution is active
  -> load conversation history and live inventory
  -> derive or preserve system instruction
  -> invoke checkpointed graph
  -> persist truthful terminal run outcome
  -> append the user/assistant turn idempotently by run_id
  -> mark history committed or leave it for deterministic reconciliation
  -> release lease
  -> return response and resume token
```

Conversation continuity, durable run identity, and LangGraph execution-thread identity are separate concepts. The local gate is not a distributed correctness boundary. With `RUN_REPOSITORY_BACKEND=postgres`, an expiring PostgreSQL lease and monotonically increasing fencing token prevent stale owners from committing after takeover.

## Durable run status

Canonical statuses are:

```text
pending
running
interrupted
completed
failed
cancelled
expired
reconciling
```

Only an explicitly interrupted run may be resumed. A terminal response is replayed for the same run and request hash after restart. Request-hash, state-schema, checkpoint namespace, token version, and token signature mismatches fail closed.

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

## Model and MCP paths

Model selection remains role-based and constrained by live Ollama inventory. The current execution budget records logical model operations rather than every physical retry/fallback attempt; authoritative physical metering belongs to observability hardening.

The inventory service loads live MCP tools. The runtime router ranks compatible read-only tools and `ToolExecutor` enforces side-effect policy and budget checks. Caller metadata does not authorize write actions.

## Evidence lifecycle

Retrieved results are normalized into typed evidence records with IDs, run/task/query metadata, content hashes, trust class, timestamps, truncation, freshness, and quality fields. Full claim-grounding and artifact-lifecycle hardening belong to later stages.

## Persistence lifecycle

The persistence authorities are intentionally distinct:

- conversation store: user-visible history;
- run repository: status, lease, fencing, terminal response, and reconciliation state;
- LangGraph checkpointer: resumable graph state;
- artifact store: large external objects when enabled.

The run outcome is written before conversation history. If the process stops between those operations, the next idempotent retry replays the terminal response, calls the store's `append_turn()` method, and marks `history_committed_at`. Memory, Redis, and PostgreSQL stores implement run-scoped turn deduplication; PostgreSQL uses a unique `(thread_id, run_id, message_kind)` index.

## Resume-token lifecycle

Version 2 tokens include a key ID, request-hash version, state-schema version, checkpoint namespace, run identity, token version, issued time, and expiry. `RESUME_TOKEN_KEYS_JSON` permits old verification keys to remain during rotation while `RESUME_TOKEN_ACTIVE_KEY_ID` selects the signing key. Incrementing a run's durable token version revokes previously issued tokens.

## Streaming lifecycle

The SSE implementation still starts a background invocation, emits periodic generic work events, and cancels/awaits the task during generator cleanup. This is cancellation-aware polling, not graph-update or token streaming.

## Deployment lifecycle

```text
GitHub-hosted runner: checkout -> install -> compile -> Ruff -> deterministic pytest
self-hosted runner: checkout -> compose validation -> rollback point -> build -> deploy
post-deployment: smoke -> API checks -> backup -> cleanup or rollback -> logs
```

The image is rebuilt on the production runner and is not yet promoted by immutable digest from the test job.

## Configuration register

| Configuration item | Current behavior |
|---|---|
| `thread_id` | Public request alias for `conversation_id` |
| `execution-state-v1` checkpoint namespace | Canonical namespace for newly created execution state |
| `RESUME_TOKEN_SECRET` | Single-key signing configuration when no key-ring JSON is configured |
| memory run repository | Process-local development and degraded-operation backend |
