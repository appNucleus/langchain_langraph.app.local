#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "== Compile Python sources =="
python -m compileall -q app tests

echo "== Verify application import and version source =="
python - <<'PY'
from app import __version__
from app.factory import create_app
from app.observability import InMemoryMetrics, MetricsRegistry, metrics
from app.settings import Settings

assert __version__
assert "app_version" not in Settings.model_fields
assert "mcp_client_version" not in Settings.model_fields
assert InMemoryMetrics is MetricsRegistry
assert metrics.snapshot() == {"counters": {}, "timings": {}}
app = create_app(settings=Settings(llm_backend="echo", mcp_enabled=False, _env_file=None))
assert app.version == __version__
print(f"Application import passed; version={__version__}")
PY

echo "== Reject duplicate runtime version settings =="
if grep -RInE '^[[:space:]]*(APP_VERSION|MCP_CLIENT_VERSION)=' .env.example app 2>/dev/null; then
  echo "Duplicate runtime version setting found." >&2
  exit 1
fi

echo "== Run tests =="
python -m pytest -q

echo "== Validate Compose =="
docker compose --env-file "${DEPLOY_ENV_FILE:-.env}" config >/dev/null

echo "Phase 3 verification passed."
