# LangChain LangGraph FastAPI App

FastAPI + LangGraph application server for a local Ollama and MCP stack. The service performs bounded planning, evidence retrieval, work, verification, revision, synthesis, and final verification while keeping conversation identity separate from one graph execution.

The Docker image name is:

```text
langchain_langraph.app:release
```

The app exposes container port `8000` and maps it to host port `8001` by default:

```text
127.0.0.1:8001 -> container:8000
```

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

Inventory, metrics, chat, and streaming routes are protected when `API_KEY` is configured. Liveness is process-only. Readiness evaluates required Ollama, MCP, persistence, and artifact dependencies.

## Current graph

The compiled graph uses these stable node names:

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

The planner decomposes the request, the runtime router selects only live models and schema-compatible read-only tools, the verifier controls revision/research/replanning, and multi-task synthesis can be checked by a separate final verifier.

See [`docs/architecture.md`](docs/architecture.md) for the request, model, tool, evidence, persistence, and deployment lifecycles.

## Request identity and compatibility

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

The server generates conversation and run IDs when omitted. Resume tokens are signed and request-bound. Reserved identity metadata is controlled by the server.

Current limitation: same-conversation exclusion and completed-response caching are process-local. Multi-worker correctness requires a durable lease and run repository and is not claimed by the current release.

## Request examples

Minimal OpenAPI/default request:

```json
{
  "message": "Continue the analysis"
}
```

The full optional-field contract is stored at:

```text
docs/example_request/chat-complete.json
```

## Streaming status

`/api/chat/stream` is cancellation-aware and emits request/planning/working/completed events. The current implementation polls a background graph invocation; it is **not** LangGraph update streaming or Ollama token streaming. Do not describe it as real token streaming until the streaming stage is implemented and tested.

## Run locally

```bash
cp .env.example .env
docker compose --env-file .env up --build
```

Open:

```text
http://127.0.0.1:8001/health
```

Check live inventory:

```bash
curl http://127.0.0.1:8001/api/inventory
```

Test chat:

```bash
curl -X POST http://127.0.0.1:8001/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is the weather in Indianapolis tomorrow?"}'
```

## Agent-loop configuration

Use the canonical domain-oriented settings:

```env
AGENT_MAX_ITERATIONS=4
AGENT_MAX_RESEARCH_ROUNDS=2
AGENT_MAX_REPLANS=1
AGENT_MAX_CONTEXT_CHARS=16000
```

The former `PHASE2_*` environment names remain accepted as deprecated aliases. New code and deployment configuration must use `AGENT_*` names.

`RUN_CHECKPOINT_NAMESPACE=phase5-v1` is deliberately retained for existing checkpoint compatibility. Changing it requires an explicit checkpoint migration or a fail-closed version transition.

## Ollama configuration

```env
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://ollama.home.arpa:11434
```

Model roles are configured independently. `/api/inventory` reports configured roles and their live resolution against Ollama inventory.

## MCP configuration

```env
MCP_ENABLED=true
MCP_SERVER_URL=https://mcp.home.arpa/mcp
MCP_VERIFY_TLS=false
MCP_FOLLOW_REDIRECTS=true
```

The runtime loads `tools/list` and ranks only compatible tools reported by that live inventory. Read-only tools are allowed by policy; write-capable, ambiguous, and unknown tools fail closed unless a later server-side authorization protocol explicitly enables them.

## Persistence

Supported combinations:

```text
Conversation history: memory, Redis, or PostgreSQL
LangGraph checkpoints: memory or PostgreSQL
Artifacts: disabled or MinIO
```

Optional dependency startup failures enter explicit degraded memory mode. Required dependency failures stop startup. MinIO availability does not by itself prove normal evidence-flow integration.

## Deployment and rollback

Local deployment:

```bash
./scripts/deploy-local.sh
```

GitHub Actions deployment:

```text
.github/workflows/deploy-release.yml
```

The workflow compiles, lints, tests, builds, deploys, performs smoke/API checks, creates a known-good backup, and rolls back failures. Current limitation: the production runner rebuilds the candidate image after source tests, so the workflow does not yet prove that one immutable image digest was built once, tested, and deployed unchanged.

## API key protection

By default, `API_KEY=` is empty. For any network exposure, set a long random value and pass it as `X-API-Key`.

## Tests

Deterministic quality gates:

```bash
python -m pip check
python -m compileall -q app tests
python -m ruff check app tests
python -m pytest -v -ra --tb=long -W default -m "not live_integration"
docker compose --env-file .env.example config
```

Convenience verification:

```bash
bash ./scripts/verify-runtime.sh
```

Live integration tests remain separately tagged and require reachable Ollama and MCP services.
