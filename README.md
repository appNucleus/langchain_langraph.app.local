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

## API-layer organization

The FastAPI entry point remains `app.main:app`, and `create_app(settings=None, chat_agent=None)` remains the application factory and test seam. HTTP responsibilities are separated by role:

```text
app/
├── factory.py                 # application composition only
├── main.py                    # ASGI entry point
├── core/
│   ├── __init__.py
│   └── lifespan.py            # startup and shutdown lifecycle
└── api/
    ├── __init__.py
    ├── router.py              # single router composition point
    ├── dependencies.py        # API-key dependency
    ├── exception_handlers.py  # exception registration and mappings
    ├── openapi.py             # lazy Swagger example customization
    └── routes/
        ├── __init__.py
        ├── root.py
        ├── health.py
        ├── inventory.py
        ├── metrics.py
        └── chat.py
```

To follow an HTTP request, start at `app/main.py`, then `app/factory.py`, `app/api/router.py`, and the matching module under `app/api/routes/`. Graph execution, agents, persistence, model/tool clients, and schemas remain outside the API layer.

`app/factory.py` is intentionally limited to composition: settings and logging, agent construction, FastAPI construction, `app.state`, middleware, exception-handler registration, root-router inclusion, and OpenAPI customization. It does not own route handlers, lifecycle implementation, database clients, or LLM clients.

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

## Swagger/OpenAPI request examples

The two POST operations use separate documentation-only request examples:

```text
docs/example_request/chat.json         -> POST /api/chat
docs/example_request/chat-stream.json  -> POST /api/chat/stream
```

These files are read lazily only while OpenAPI is generated, such as for `/openapi.json`, `/docs`, or `/redoc`. They are validated against `ChatRequest` before insertion into the schema.

They are not:

- Pydantic field defaults;
- runtime request defaults or fallback request data;
- chat, graph, agent, persistence, database, model, or tool input;
- application startup or configuration input;
- pytest fixture data.

Missing or invalid documentation JSON does not prevent application startup or ordinary API execution. Tests inject code-defined examples through the preserved `app.factory.load_chat_request_example` test seam, so the test suite does not depend on production files under `docs/example_request/`.

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

### Connection lifecycle and dbs.local alignment

The application owns one persistence runtime per FastAPI application process. PostgreSQL conversation history and durable run records reuse one bounded `asyncpg` pool instead of creating independent pools. The LangGraph PostgreSQL checkpointer uses one separate bounded application-scoped `psycopg` pool because its driver is not compatible with the `asyncpg` pool. Redis uses one bounded application client and MinIO uses one application client.

The example endpoints and credentials are aligned with `appNucleus/dbs.local`:

```env
DATABASE_URL=postgresql://langgraph_user:change_me_postgres_2026@dbs.home.arpa:5432/langgraph_app
REDIS_URL=redis://default:change_me_redis_2026@dbs.home.arpa:6379/0
REDIS_MAX_CONNECTIONS=20
MINIO_ENDPOINT=dbs.home.arpa:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=change_me_minio_2026
MINIO_BUCKET=langgraph-app
```

`pgAdmin` and `RedisInsight` are administration interfaces, not application persistence backends. The database stack also provides Neo4j, but this application has no active Neo4j runtime path, so no unused Neo4j client or connection is created.

`/health/ready` includes non-secret connection-management metadata such as whether the shared PostgreSQL pool is active and its bounded size.

## Execution usage metadata

Successful and safely terminated chat responses expose the complete serialized execution-meter snapshot under `metadata.usage`. Compatibility fields such as `model_calls`, `tool_calls`, and `elapsed_seconds` remain present alongside physical attempt, failure, fallback, token, queue, deadline, checkpoint, artifact, and cancellation counters.

## Deployment and rollback

The release workflow compiles, lints, tests, builds, deploys, performs smoke/API checks, creates a known-good backup, and rolls back failures. It still rebuilds the image on the production runner; immutable build-once/deploy-by-digest remains a later-stage objective.

## Tests

```bash
python -m pip check
python -m compileall -q app tests scripts/check_coverage_thresholds.py
python -m ruff check app tests scripts/check_coverage_thresholds.py
python -m pytest -v -ra --tb=long -W default -m "not live_integration"
python -m coverage xml --rcfile=pyproject.toml -o test-results/coverage.xml
python -m coverage json --rcfile=pyproject.toml -o test-results/coverage.json
python -m coverage report --rcfile=pyproject.toml
python scripts/check_coverage_thresholds.py test-results/coverage.json --config pyproject.toml
docker compose --env-file .env.example config
```

Live PostgreSQL, Ollama, MCP, Redis, and MinIO tests remain separately tagged and require reachable services.
