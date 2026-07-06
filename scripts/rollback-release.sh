#!/usr/bin/env bash
set -Eeuo pipefail

: "${BACKUP_ROOT:?BACKUP_ROOT is required}"
: "${DEPLOY_ENV_FILE:?DEPLOY_ENV_FILE is required}"
: "${ROLLBACK_STATE_DIR:?ROLLBACK_STATE_DIR is required}"
: "${ROLLBACK_IMAGE:?ROLLBACK_IMAGE is required}"
: "${RELEASE_IMAGE:?RELEASE_IMAGE is required}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set_github_result() {
  local result="$1"
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    echo "ROLLBACK_RESULT=$result" >> "$GITHUB_ENV"
  fi
}

if [[ ! -f "$ROLLBACK_STATE_DIR/has-rollback" ]] || \
   [[ "$(cat "$ROLLBACK_STATE_DIR/has-rollback")" != "true" ]]; then
  echo "::error::Deployment failed and no previous running deployment was available for rollback."
  set_github_result "unavailable"
  exit 0
fi

if ! docker image inspect "$ROLLBACK_IMAGE" >/dev/null 2>&1; then
  echo "::error::Deployment failed, but the prepared rollback image no longer exists."
  set_github_result "failed"
  exit 1
fi

echo "::group::Failed candidate diagnostics"
set +e
docker compose --env-file "$DEPLOY_ENV_FILE" ps --all
docker compose --env-file "$DEPLOY_ENV_FILE" logs --no-color --tail=200
set -e
echo "::endgroup::"

if [[ -f "$ROLLBACK_STATE_DIR/new-backup-path.txt" ]]; then
  candidate_backup="$(cat "$ROLLBACK_STATE_DIR/new-backup-path.txt")"
  rm -rf -- "$candidate_backup"
fi

if [[ -d "$ROLLBACK_STATE_DIR/previous-backup" && \
      -f "$ROLLBACK_STATE_DIR/previous-backup-path.txt" ]]; then
  previous_backup_name="$(basename "$(cat "$ROLLBACK_STATE_DIR/previous-backup-path.txt")")"
  previous_backup_destination="$BACKUP_ROOT/$previous_backup_name"
  if [[ ! -d "$previous_backup_destination" ]]; then
    mkdir -p "$BACKUP_ROOT"
    cp -a "$ROLLBACK_STATE_DIR/previous-backup" "$previous_backup_destination"
    echo "Restored previous successful backup: $previous_backup_destination"
  fi
fi

install -m 600 "$ROLLBACK_STATE_DIR/runtime.env" "$DEPLOY_ENV_FILE"
docker image tag "$ROLLBACK_IMAGE" "$RELEASE_IMAGE"

pushd "$ROLLBACK_STATE_DIR/source" >/dev/null
docker compose --env-file "$ROLLBACK_STATE_DIR/runtime.env" up \
  --detach \
  --no-build \
  --force-recreate \
  --remove-orphans \
  --wait \
  --wait-timeout 120
popd >/dev/null

"$script_dir/smoke-test.sh" rollback

docker image rm "$ROLLBACK_IMAGE" >/dev/null 2>&1 || true
docker image prune --force >/dev/null || true

set_github_result "success"
echo "::error::Candidate deployment failed. The last successful deployment was restored automatically."
echo "Rollback completed successfully. The GitHub workflow intentionally remains failed."
