#!/usr/bin/env bash
set -Eeuo pipefail

: "${DEPLOY_ENV_FILE:?DEPLOY_ENV_FILE is required}"

mode="${1:-candidate}"
container_name="${CONTAINER_NAME:-langchain-langraph-app}"

read_env_value() {
  local key="$1"
  local value
  value="$(
    awk -F= -v wanted="$key" '
      /^[[:space:]]*#/ { next }
      $1 == wanted {
        sub(/^[^=]*=/, "")
        print
      }
    ' "$DEPLOY_ENV_FILE" | tail -n 1
  )"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  printf '%s' "$value"
}

app_port="$(read_env_value APP_PORT)"
app_port="${app_port:-8001}"

if [[ ! "$app_port" =~ ^[0-9]+$ ]] || (( app_port < 1 || app_port > 65535 )); then
  echo "Invalid APP_PORT in $DEPLOY_ENV_FILE: $app_port" >&2
  exit 1
fi

run_fastapi_smoke() {
  docker exec "$container_name" python - <<'PY'
from app.main import app
print("container import OK")
print(app.title)
PY

  curl --fail --silent --show-error --max-time 10 \
    "http://127.0.0.1:${app_port}/health" >/dev/null
}

case "$mode" in
  candidate)
    run_fastapi_smoke
    echo "Candidate FastAPI smoke test passed: http://127.0.0.1:${app_port}/health"
    ;;
  rollback)
    run_fastapi_smoke
    echo "Rollback FastAPI smoke test passed: http://127.0.0.1:${app_port}/health"
    ;;
  *)
    echo "Unknown smoke-test mode: $mode" >&2
    exit 2
    ;;
esac
