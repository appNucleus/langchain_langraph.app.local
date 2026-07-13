#!/usr/bin/env bash
set -euo pipefail

python -m pip check
python -m compileall -q app tests
python -m ruff check app tests
python -m pytest -v -ra --tb=long -W default -m "not live_integration"

python - <<'PY'
from app import __version__
from app.factory import create_app
from app.settings import Settings

settings = Settings(_env_file=None, llm_backend="echo", mcp_enabled=False)
app = create_app(settings=settings)

assert app.version == __version__
assert hasattr(settings, "ollama_max_concurrency")
assert hasattr(settings, "ollama_max_concurrent_requests")
assert hasattr(settings, "mcp_read_timeout_seconds")
assert hasattr(settings, "inventory_cache_ttl_seconds")
assert hasattr(settings, "agent_max_research_rounds")
assert settings.phase2_max_research_rounds == settings.agent_max_research_rounds
print(f"Import/startup contract OK; version={__version__}")
PY

if command -v docker >/dev/null 2>&1 && [[ -f compose.yaml ]]; then
  ENV_FILE="${DEPLOY_ENV_FILE:-.env.example}"
  docker compose --env-file "$ENV_FILE" config >/dev/null
  echo "Docker Compose configuration OK"
else
  echo "Skipping Compose validation: Docker or compose.yaml is unavailable"
fi
