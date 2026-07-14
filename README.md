# LangChain LangGraph FastAPI App

FastAPI + LangGraph application server for a local Ollama and MCP stack. The service performs bounded planning, evidence retrieval, work, verification, revision, synthesis, and final verification while keeping conversation continuity separate from one graph execution.

The Docker image is `langchain_langraph.app:release`; container port `8000` maps to `127.0.0.1:8001` by default.

## Current API

```text
GET  /
GET  /health
GET  /health/live
GET  /health/ready
GET  /api/inventory
GET  /api/metrics
POST /api/chat
POST /api/chat/stream
```

Inventory, metrics, chat, and streaming routes are protected when `API_KEY` is configured. Liveness is process-only. Readiness evaluates required Ollama, MCP, conversation, run-repository, checkpoint, and artifact dependencies.

## Current graph

```text
START -> plan
plan -> research | worker | terminate
research -> worker | terminate
worker -> verify | terminate
verify -> advance | revise | research | replan | terminate
revise -> verify | terminate
replan -> worker | terminate
advance -> research | worker | finalize
finalize -> END | verify_final | terminate
verify_final -> END | revise_final | terminate
revise_final -> verify_final
terminate -> END
```

See [`docs/architecture.md`](docs/architecture.md) for the request, run, model, tool, evidence, persistence, and deployment lifecycles.

## Request identity and durable outcomes

The public request supports:

```text
message
thread_id              # deprecated compatibility alias for conversation_id
conversation_id
run_id
resume
resume_token
system_prompt
metadata
```

The server generates conversation and run IDs when omitted. Each run uses a distinct LangGraph execution thread. Resume tokens are signed, request-bound, versioned, and may use a rotatable key ring. Reserved identity metadata remains server controlled.

`RUN_REPOSITORY_BACKEND=memory` preserves the lightweight local default. Set `RUN_REPOSITORY_BACKEND=postgres` with `DATABASE_URL` to enable:

- restart-safe completed/failed response replay;
- distributed same-conversation leases;
- lease renewal and fencing against stale owners;
- durable run status and termination metadata;
- resume-token revocation versions;
- explicit history reconciliation after a crash.

The process-local conversation gate remains a fast-path optimization only. PostgreSQL is the correctness boundary when the PostgreSQL run repository is enabled.

## Request examples

The Swagger/OpenAPI request body is documentation input loaded from:

```text
docs/example_request/chat.json
```

This file is loaded lazily only when OpenAPI is generated, such as for
`/openapi.json`, `/docs`, or `/redoc`. Its contents do not define
`ChatRequest` model defaults, are not supplied to chat execution, and are not
used as pytest fixture data. Tests inject code-defined request examples so
documentation can evolve without changing runtime or test semantics.

An additional documentation example may be maintained at:

```text
docs/example_request/chat-complete.json
```

Neither documentation file is a runtime request template.

## Streaming status

`/api/chat/stream` remains cancellation-aware polling that emits request/planning/working/completed events. It is not yet LangGraph update streaming or Ollama token streaming; that belongs to a later stage.

## Run locally

```bash
cp .env.example .env
docker compose --env-file .env up --build
```

```bash
curl -X POST http://127.0.0.1:8001/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is the weather in Indianapolis tomorrow?"}'
```

## Run identity configuration

Single-key compatibility:

```env
RESUME_TOKEN_SECRET=replace-with-a-long-random-secret
RESUME_TOKEN_ACTIVE_KEY_ID=primary
```

Rotatable key ring:

```env
RESUME_TOKEN_ACTIVE_KEY_ID=2026-07-primary
RESUME_TOKEN_KEYS_JSON={"2026-07-primary":"new-secret","2026-06-previous":"old-secret"}
```

Durable run ownership:

```env
RUN_REPOSITORY_BACKEND=postgres
RUN_LEASE_TTL_SECONDS=30
RUN_LEASE_HEARTBEAT_SECONDS=10
RUN_REQUEST_HASH_VERSION=1
```

Use `RUN_CHECKPOINT_NAMESPACE=execution-state-v1` for newly created execution checkpoints.

## Agent-loop configuration

```env
AGENT_MAX_ITERATIONS=4
AGENT_MAX_RESEARCH_ROUNDS=2
AGENT_MAX_REPLANS=1
AGENT_MAX_CONTEXT_CHARS=16000
```

Only the canonical `AGENT_*` names are supported.

## Persistence

Supported combinations:

```text
Conversation history: memory, Redis, or PostgreSQL
Run outcomes/leases: memory or PostgreSQL
LangGraph checkpoints: memory or PostgreSQL
Artifacts: disabled or MinIO
```

For durable multi-process execution use PostgreSQL for the run repository and checkpoints. Conversation turns are appended idempotently by `run_id`; PostgreSQL and Redis enforce that independently of the bounded visible history window. Optional dependency startup failures enter explicit memory degradation; required dependency failures stop startup.

## Deployment and rollback

The release workflow compiles, lints, tests, builds, deploys, performs smoke/API checks, creates a known-good backup, and rolls back failures. It still rebuilds the image on the production runner; immutable build-once/deploy-by-digest remains a later-stage objective.

## Tests

```bash
python -m pip check
python -m compileall -q app tests
python -m ruff check app tests
python -m pytest -v -ra --tb=long -W default -m "not live_integration"
docker compose --env-file .env.example config
```

Live PostgreSQL, Ollama, MCP, Redis, and MinIO tests remain separately tagged and require reachable services.
