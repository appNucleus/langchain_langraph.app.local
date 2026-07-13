#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ENV_FILE="${DEPLOY_ENV_FILE:-$HOME/.config/langchain-langraph-app/runtime.env}"
if [[ ! -f "$DEPLOY_ENV_FILE" ]]; then
  echo "Deployment environment file not found: $DEPLOY_ENV_FILE" >&2
  exit 1
fi

read_env_value() {
  local key="$1"
  sed -n "s/^${key}=//p" "$DEPLOY_ENV_FILE" | tail -n 1
}

APP_PORT="$(read_env_value APP_PORT)"
APP_PORT="${APP_PORT:-8001}"
API_KEY="$(read_env_value API_KEY)"
BASE_URL="http://127.0.0.1:${APP_PORT}"
AUTH_ARGS=()
if [[ -n "$API_KEY" ]]; then
  AUTH_ARGS=(-H "X-API-Key: $API_KEY")
fi

request_json() {
  local method="$1" url="$2" body="${3:-}"
  if [[ -n "$body" ]]; then
    curl --fail --silent --show-error --max-time 210 \
      -X "$method" "${AUTH_ARGS[@]}" -H 'Content-Type: application/json' \
      --data "$body" "$url"
  else
    curl --fail --silent --show-error --max-time 30 \
      -X "$method" "${AUTH_ARGS[@]}" "$url"
  fi
}

validate_json() {
  local expression="$1"
  python3 -c "import json,sys; data=json.load(sys.stdin); assert ${expression}"
}

request_json GET "$BASE_URL/health/live" | validate_json "data.get('status') == 'alive'"
request_json GET "$BASE_URL/health/ready" | validate_json "data.get('status') == 'ready'"
request_json GET "$BASE_URL/api/inventory" | validate_json "isinstance(data, dict) and bool(data)"

THREAD_ID="deployment-integration-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}"
CHAT_BODY=$(python3 - "$THREAD_ID" <<'PY'
import json, sys
print(json.dumps({
    "message": "Reply with one brief greeting. Do not use external tools.",
    "thread_id": sys.argv[1],
    "system_prompt": "Use no tools. Return a concise greeting.",
    "metadata": {"safe_read_only_test": True, "deployment_integration": True},
}))
PY
)
request_json POST "$BASE_URL/api/chat" "$CHAT_BODY" | validate_json "isinstance(data.get('response'), str) and bool(data['response'].strip()) and data.get('thread_id')"

echo "Post-deployment API integration test passed."
