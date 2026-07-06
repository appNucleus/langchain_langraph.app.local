#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export CONTAINER_NAME="${CONTAINER_NAME:-langchain-langraph-app}"
export RELEASE_IMAGE="${RELEASE_IMAGE:-langchain_langraph.app:release}"
export ROLLBACK_IMAGE="${ROLLBACK_IMAGE:-langchain_langraph.app:rollback}"
export BACKUP_ROOT="${BACKUP_ROOT:-$HOME/backup_app.local}"
export DEPLOY_ENV_FILE="${DEPLOY_ENV_FILE:-$HOME/.config/langchain-langraph-app/runtime.env}"
export ROLLBACK_STATE_DIR="${ROLLBACK_STATE_DIR:-${TMPDIR:-/tmp}/langchain-langraph-app-rollback-$USER}"
export GITHUB_SHA="${GITHUB_SHA:-$(git rev-parse HEAD)}"
export GITHUB_RUN_ID="${GITHUB_RUN_ID:-manual-$(date +%s)}"
export GITHUB_RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"

mkdir -p "$(dirname "$DEPLOY_ENV_FILE")"
if [[ ! -f "$DEPLOY_ENV_FILE" ]]; then
  install -m 600 .env.example "$DEPLOY_ENV_FILE"
else
  chmod 600 "$DEPLOY_ENV_FILE"
fi

set_runtime_env_value() {
  local key="$1"
  local value="$2"
  local file="$3"
  if grep -qE "^${key}=" "$file"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "$file"
  else
    printf '\n%s=%s\n' "$key" "$value" >> "$file"
  fi
}

# Enforce this server's intended Caddy-facing bind.
set_runtime_env_value HOST_BIND 127.0.0.1 "$DEPLOY_ENV_FILE"
set_runtime_env_value APP_PORT 8001 "$DEPLOY_ENV_FILE"

rollback_on_error() {
  local original_status=$?
  trap - ERR
  echo "Local deployment failed with exit code $original_status. Attempting rollback..." >&2
  bash ./scripts/rollback-release.sh || true
  exit "$original_status"
}
trap rollback_on_error ERR

bash ./scripts/verify-server.sh

docker compose --env-file "$DEPLOY_ENV_FILE" config >/dev/null
bash ./scripts/prepare-rollback.sh

docker compose --env-file "$DEPLOY_ENV_FILE" build --pull

docker compose --env-file "$DEPLOY_ENV_FILE" up \
  --detach \
  --no-build \
  --force-recreate \
  --remove-orphans \
  --wait \
  --wait-timeout 120

bash ./scripts/smoke-test.sh candidate
bash ./scripts/create-success-backup.sh

docker image rm "$ROLLBACK_IMAGE" >/dev/null 2>&1 || true
if ! docker image prune --force; then
  echo "Warning: deployment succeeded, but Docker image cleanup failed." >&2
fi

docker compose --env-file "$DEPLOY_ENV_FILE" ps

trap - ERR
echo "Deployment completed successfully."
