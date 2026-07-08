# LangChain LangGraph FastAPI App

FastAPI + LangGraph application server for your local assistant stack. It is designed to sit beside your existing MCP server and Ollama server, then choose the best available local model and MCP tools per request.

The Docker image name is intentionally set to:

```text
langchain_langraph.app:release
```

The app exposes container port `8000` and maps it to host port `8001` by default:

```text
127.0.0.1:8001 -> container:8000
```

## What this includes

- FastAPI endpoints: `/health`, `/health/live`, `/api/inventory`, `/api/chat`, and `/api/chat/stream`.
- LangGraph workflow: `prepare -> live inventory -> query planning -> query rewrite -> MCP tools -> final answer`.
- Live inventory loading from:
  - Ollama `GET /api/tags` for local model list.
  - MCP `tools/list` for tool list.
- Dynamic model role selection with fallback if a configured model is not currently available.
- Compound-query decomposition into simple subqueries, with per-subquery model/tool routing.
- Web/news/search-style MCP fallback for fresh or unknown topics when those tools are available.
- Final synthesis for multi-part questions.
- Clean answer prompt that prefers broad, useful, easy-to-read answers and separate references.
- Safe email-send guard: natural language alone cannot send a draft.
- Docker Compose deployment with host bind controlled by `HOST_BIND` and `APP_PORT`.
- GitHub Actions release deployment using the same rollback/backup style as your `mcp.local` repo.
- Local deployment script: `./scripts/deploy-local.sh`.

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
  -d '{"message":"What is the latest OpenAI news? Also what is the weather in Indianapolis tomorrow?"}'
```

## Ollama configuration

Default local-network Ollama URL:

```env
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://ollama.home.arpa:11434
```

Model roles are configured separately so the app can balance quality and speed:

```env
MODEL_PLANNER=qwen3.5:4b
MODEL_SIMPLE=qwen3.5:2b
MODEL_GENERAL=qwen3.5:4b
MODEL_SEARCH=qwen3.5:9b
MODEL_REASONING=phi4-mini-reasoning:latest
MODEL_HEAVY=gemma4:26b-a4b-it-qat
MODEL_SYNTHESIS=gemma4:26b-a4b-it-qat
MODEL_VISION=qwen3-vl:4b
MODEL_FALLBACK=qwen3.5:4b
EMBEDDING_MODEL=qwen3-embedding:0.6b
```

The `/api/inventory` endpoint shows both configured roles and the resolved live model chosen from Ollama.

## MCP configuration

Default local-network MCP URL:

```env
MCP_ENABLED=true
MCP_SERVER_URL=https://mcp.home.arpa/mcp
MCP_VERIFY_TLS=false
MCP_FOLLOW_REDIRECTS=true
```

The app calls `tools/list` first, then only plans against tools that the MCP server reports as live. If a preferred tool is unavailable, it tries a safe fallback such as `web_search_and_scrape -> web_search -> news_search`.

## Deployment using rollback flow

```bash
./scripts/deploy-local.sh
```

This enforces:

```env
HOST_BIND=127.0.0.1
APP_PORT=8001
```

and stores the persistent runtime env at:

```text
~/.config/langchain-langraph-app/runtime.env
```

## GitHub Actions deployment

The workflow deploys when you push to the `release` branch:

```text
.github/workflows/deploy-release.yml
```

It expects a self-hosted runner with labels:

```text
self-hosted, Linux, X64, app-prod
```

Change the runner label in the workflow if your runner uses a different label.

## Caddy example

For a local/private domain:

```caddyfile
langchain.home.arpa {
    tls internal
    reverse_proxy 127.0.0.1:8001
}
```

For a public HTTPS domain, keep the app bound to `127.0.0.1:8001` and expose only Caddy on ports `80` and `443`.

## API key protection

By default, `API_KEY=` is empty, so `/api/chat` and `/api/inventory` are reachable by anyone who can reach the service.

For public use, set:

```env
API_KEY=change-me-long-random-value
```

Then call:

```bash
curl -X POST http://127.0.0.1:8001/api/chat \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: change-me-long-random-value' \
  -d '{"message":"hello"}'
```

## Tests

```bash
python -m pytest -q
```

Live integration tests are skipped by default. To run them against your local network services:

```bash
LIVE_INTEGRATION=1 \
OLLAMA_BASE_URL=http://ollama.home.arpa:11434 \
MCP_SERVER_URL=https://mcp.home.arpa/mcp \
MCP_VERIFY_TLS=false \
python -m pytest -q tests/test_live_integration.py
```
