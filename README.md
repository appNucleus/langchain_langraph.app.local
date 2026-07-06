# LangChain LangGraph FastAPI App

Minimal FastAPI + LangGraph application server designed to sit beside your existing MCP server, Ollama server, and database server.

The Docker image name is intentionally set to:

```text
langchain_langraph.app:release
```

The app exposes container port `8000` and maps it to host port `8001` by default:

```text
127.0.0.1:8001 -> container:8000
```

## What this includes

- FastAPI app with `/health`, `/api/chat`, and `/api/chat/stream`.
- Minimal LangGraph workflow: `START -> assistant -> END`.
- Safe default `LLM_BACKEND=echo`, so the first deployment can pass without Ollama.
- Ollama-ready config using `langchain-ollama`.
- MCP/database/Redis placeholders in `.env.example` for later extension.
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

Test chat:

```bash
curl -X POST http://127.0.0.1:8001/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"hello"}'
```

Expected first response is echo mode:

```text
Echo mode is active. Your FastAPI + LangGraph service is running.
```

## Enable Ollama later

Edit `.env` or the persistent runtime env file on the server:

```env
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=qwen2.5:0.5b
```

`compose.yaml` includes this mapping so Linux Docker containers can reach services on the Docker host:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

If Ollama runs on a different server, use that server's LAN URL instead, for example:

```env
OLLAMA_BASE_URL=http://192.168.1.126:11434
```

## Deploy locally using rollback flow

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

For a public HTTPS domain, still keep the app bound to `127.0.0.1:8001` and expose only Caddy on ports `80` and `443`.

## API key protection

By default, `API_KEY=` is empty, so `/api/chat` is reachable by anyone who can reach the service.

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

## Extension path

Recommended next files/modules to add later:

```text
app/tools/mcp_client.py       # call your MCP server
app/db/session.py             # PostgreSQL / SQLAlchemy connection
app/db/models.py              # users, threads, messages, tool_calls
app/rag/vector_store.py       # pgvector or Qdrant
app/graph.py                  # expand graph nodes and routing
```

The current graph is deliberately small so you can extend it cleanly.
