#!/usr/bin/env bash

set -euo pipefail

DEPLOY_ENV_FILE="${DEPLOY_ENV_FILE:-$HOME/.config/langchain-langraph-app/runtime.env}"

if [[ ! -f "$DEPLOY_ENV_FILE" ]]; then
  echo "::error title=Post-deployment integration setup failed::Deployment environment file not found: $DEPLOY_ENV_FILE" >&2
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

TEST_TOTAL=0
TEST_PASSED=0
TEST_FAILED=0
PASSED_TESTS=()
FAILED_TESTS=()

print_redacted_json() {
  local response_file="$1"
  local max_chars="${2:-6000}"

  python3 - "$response_file" "$max_chars" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
max_chars = int(sys.argv[2])
text = path.read_text(encoding="utf-8", errors="replace")

if not text.strip():
    print("<empty response body>")
    raise SystemExit(0)

try:
    data = json.loads(text)
except json.JSONDecodeError:
    rendered = text
else:
    sensitive_parts = (
        "api_key",
        "apikey",
        "authorization",
        "password",
        "secret",
        "token",
        "credential",
    )

    def redact(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                normalized = str(key).lower().replace("-", "_")
                if any(part in normalized for part in sensitive_parts):
                    result[key] = "<redacted>"
                else:
                    result[key] = redact(item)
            return result
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    rendered = json.dumps(redact(data), indent=2, ensure_ascii=False, sort_keys=True)

if len(rendered) > max_chars:
    omitted = len(rendered) - max_chars
    rendered = f"{rendered[:max_chars]}\n... <truncated {omitted} characters>"

print(rendered)
PY
}

print_api_diagnostics() {
  local response_file="$1"

  python3 - "$response_file" <<'PY'
import json
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
try:
    data = json.loads(text)
except json.JSONDecodeError:
    raise SystemExit(0)

matches = []
interesting_names = {
    "warning",
    "warnings",
    "error",
    "errors",
    "exception",
    "exceptions",
    "degraded",
    "unavailable",
    "failure",
    "failures",
}
problem_statuses = {"degraded", "error", "failed", "failure", "unavailable", "unhealthy"}


def is_meaningful(value):
    return value not in (None, "", [], {}, False, 0)


def walk(value, path="response"):
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            normalized = str(key).lower().replace("-", "_")
            if normalized in interesting_names and is_meaningful(item):
                matches.append((child, item))
            elif normalized == "status" and isinstance(item, str) and item.lower() in problem_statuses:
                matches.append((child, item))
            walk(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            walk(item, f"{path}[{index}]")


walk(data)
for path, value in matches:
    rendered = json.dumps(value, ensure_ascii=False)
    if len(rendered) > 500:
        rendered = rendered[:500] + "..."
    print(f"{path}: {rendered}")
PY
}

validate_json_file() {
  local response_file="$1"
  local expression="$2"

  python3 - "$response_file" "$expression" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expression = sys.argv[2]

try:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
except json.JSONDecodeError as exc:
    print(f"JSON validation error: response is not valid JSON: {exc}", file=sys.stderr)
    raise SystemExit(1)

scope = {
    "data": data,
    "bool": bool,
    "dict": dict,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "str": str,
}

try:
    valid = bool(eval(expression, {"__builtins__": {}}, scope))
except Exception as exc:
    print(f"Validation expression raised {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)

if not valid:
    print(f"Validation expression returned false: {expression}", file=sys.stderr)
    raise SystemExit(1)
PY
}

run_json_test() {
  local test_name="$1"
  local method="$2"
  local path="$3"
  local validation_expression="$4"
  local body="${5:-}"
  local max_time="${6:-30}"
  local response_file stderr_file metrics curl_status http_status duration size_download
  local diagnostics validation_status test_failed=0

  TEST_TOTAL=$((TEST_TOTAL + 1))
  response_file="$(mktemp)"
  stderr_file="$(mktemp)"

  echo "::group::Post-deployment test ${TEST_TOTAL}: ${test_name}"
  echo "Test:       ${test_name}"
  echo "Request:    ${method} ${path}"
  echo "Base URL:   ${BASE_URL}"
  echo "Timeout:    ${max_time}s"

  set +e
  if [[ -n "$body" ]]; then
    metrics="$(
      curl --silent --show-error \
        --connect-timeout 10 \
        --max-time "$max_time" \
        --output "$response_file" \
        --write-out $'%{http_code}\t%{time_total}\t%{size_download}' \
        -X "$method" \
        "${AUTH_ARGS[@]}" \
        -H 'Content-Type: application/json' \
        --data "$body" \
        "${BASE_URL}${path}" \
        2>"$stderr_file"
    )"
    curl_status=$?
  else
    metrics="$(
      curl --silent --show-error \
        --connect-timeout 10 \
        --max-time "$max_time" \
        --output "$response_file" \
        --write-out $'%{http_code}\t%{time_total}\t%{size_download}' \
        -X "$method" \
        "${AUTH_ARGS[@]}" \
        "${BASE_URL}${path}" \
        2>"$stderr_file"
    )"
    curl_status=$?
  fi
  set -e

  IFS=$'\t' read -r http_status duration size_download <<<"${metrics:-000\t0\t0}"

  echo "HTTP:       ${http_status:-000}"
  echo "Duration:   ${duration:-0}s"
  echo "Downloaded: ${size_download:-0} bytes"

  if [[ -s "$stderr_file" ]]; then
    echo "curl diagnostics:"
    sed 's/^/  /' "$stderr_file"
  else
    echo "curl diagnostics: none"
  fi

  echo "Response body (redacted, maximum 6000 characters):"
  print_redacted_json "$response_file" 6000 | sed 's/^/  /'

  diagnostics="$(print_api_diagnostics "$response_file")"
  if [[ -n "$diagnostics" ]]; then
    echo "API warnings/errors detected:"
    while IFS= read -r line; do
      echo "  ${line}"
      echo "::warning title=${test_name} API diagnostic::${line}"
    done <<<"$diagnostics"
  else
    echo "API warnings/errors detected: none"
  fi

  if (( curl_status != 0 )); then
    echo "::error title=${test_name} transport failure::curl exited with status ${curl_status}"
    test_failed=1
  fi

  if [[ ! "${http_status:-}" =~ ^2[0-9][0-9]$ ]]; then
    echo "::error title=${test_name} HTTP failure::Expected HTTP 2xx but received ${http_status:-000}"
    test_failed=1
  fi

  set +e
  validate_json_file "$response_file" "$validation_expression"
  validation_status=$?
  set -e

  if (( validation_status != 0 )); then
    echo "::error title=${test_name} response validation failed::The response did not satisfy the integration-test contract"
    test_failed=1
  fi

  if (( test_failed == 0 )); then
    TEST_PASSED=$((TEST_PASSED + 1))
    PASSED_TESTS+=("$test_name")
    echo "Result:     PASS"
  else
    TEST_FAILED=$((TEST_FAILED + 1))
    FAILED_TESTS+=("$test_name")
    echo "Result:     FAIL"
  fi

  rm -f "$response_file" "$stderr_file"
  echo "::endgroup::"
  echo
}

THREAD_ID="deployment-integration-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}"
CHAT_BODY="$(
  python3 - "$THREAD_ID" <<'PY'
import json
import sys

print(json.dumps({
    "message": "Reply with one brief greeting. Do not use external tools.",
    "thread_id": sys.argv[1],
    "system_prompt": "Use no tools. Return a concise greeting.",
    "metadata": {
        "safe_read_only_test": True,
        "deployment_integration": True,
    },
}))
PY
)"

run_json_test \
  "Liveness endpoint" \
  GET \
  "/health/live" \
  "data.get('status') == 'alive'"

run_json_test \
  "Readiness endpoint" \
  GET \
  "/health/ready" \
  "data.get('status') == 'ready'"

run_json_test \
  "Runtime inventory endpoint" \
  GET \
  "/api/inventory" \
  "isinstance(data, dict) and bool(data)"

run_json_test \
  "Controlled no-tool chat" \
  POST \
  "/api/chat" \
  "isinstance(data.get('response'), str) and bool(data['response'].strip()) and bool(data.get('thread_id'))" \
  "$CHAT_BODY" \
  210

echo "Post-deployment API integration summary"
echo "======================================="
echo "Base URL: ${BASE_URL}"
echo "Tests run: ${TEST_TOTAL}"
echo "Passed:    ${TEST_PASSED}"
echo "Failed:    ${TEST_FAILED}"

if (( ${#PASSED_TESTS[@]} > 0 )); then
  echo "Passed tests:"
  printf '  - %s\n' "${PASSED_TESTS[@]}"
fi

if (( ${#FAILED_TESTS[@]} > 0 )); then
  echo "Failed tests:"
  printf '  - %s\n' "${FAILED_TESTS[@]}"
  echo "::error title=Post-deployment API integration failed::${TEST_FAILED} of ${TEST_TOTAL} tests failed"
  exit 1
fi

echo "Post-deployment API integration test passed: ${TEST_PASSED}/${TEST_TOTAL} checks succeeded."
