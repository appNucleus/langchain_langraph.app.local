#!/usr/bin/env bash
set -Eeuo pipefail

: "${BACKUP_ROOT:?BACKUP_ROOT is required}"
: "${DEPLOY_ENV_FILE:?DEPLOY_ENV_FILE is required}"
: "${ROLLBACK_STATE_DIR:?ROLLBACK_STATE_DIR is required}"
: "${CONTAINER_NAME:?CONTAINER_NAME is required}"
: "${RELEASE_IMAGE:?RELEASE_IMAGE is required}"

umask 077
mkdir -p "$BACKUP_ROOT"

timestamp="$(date +'%Y%m%d-%H%M%S')"
final_backup="$BACKUP_ROOT/$timestamp"
staging_backup="$BACKUP_ROOT/.staging-${timestamp}-${GITHUB_RUN_ID:-manual}-$$"

cleanup_staging() {
  rm -rf -- "$staging_backup"
}
trap cleanup_staging EXIT

if [[ -e "$final_backup" ]]; then
  echo "Backup destination already exists: $final_backup" >&2
  exit 1
fi

mkdir -p "$staging_backup/source"

# Save all files tracked by Git at the successfully deployed commit.
# This is a compact source/configuration snapshot, not a duplicate Docker image.
git archive --format=tar HEAD | tar -xf - -C "$staging_backup/source"
install -m 600 "$DEPLOY_ENV_FILE" "$staging_backup/runtime.env"
printf '%s\n' "managed-success-backup-v1" > "$staging_backup/.langchain-langraph-success-backup"
printf '%s\n' "${GITHUB_SHA:-$(git rev-parse HEAD)}" > "$staging_backup/deployed-commit.txt"
date --iso-8601=seconds > "$staging_backup/deployed-at.txt"
printf '%s\n' "${GITHUB_RUN_ID:-manual}" > "$staging_backup/github-run-id.txt"
printf '%s\n' "${GITHUB_RUN_ATTEMPT:-1}" > "$staging_backup/github-run-attempt.txt"
printf '%s\n' "${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-local/manual}/actions/runs/${GITHUB_RUN_ID:-manual}" \
  > "$staging_backup/github-run-url.txt"

docker compose --env-file "$DEPLOY_ENV_FILE" config \
  > "$staging_backup/compose-resolved.yaml"
docker compose --env-file "$DEPLOY_ENV_FILE" ps --all \
  > "$staging_backup/docker-compose-ps.txt"
docker compose --env-file "$DEPLOY_ENV_FILE" logs --no-color --tail=200 \
  > "$staging_backup/docker-compose-logs.txt" 2>&1 || true
docker container inspect "$CONTAINER_NAME" \
  > "$staging_backup/container-inspect.json"
docker image inspect "$RELEASE_IMAGE" \
  > "$staging_backup/image-inspect.json"
docker image inspect --format='{{.Id}}' "$RELEASE_IMAGE" \
  > "$staging_backup/image-id.txt"

(
  cd "$staging_backup"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)

mv -- "$staging_backup" "$final_backup"
trap - EXIT

printf '%s\n' "$final_backup" > "$ROLLBACK_STATE_DIR/new-backup-path.txt"

# Keep exactly one successful backup.
find "$BACKUP_ROOT" \
  -mindepth 1 \
  -maxdepth 1 \
  -type d \
  ! -path "$final_backup" \
  -exec rm -rf -- {} +

if [[ -n "${GITHUB_ENV:-}" ]]; then
  echo "CURRENT_BACKUP=$final_backup" >> "$GITHUB_ENV"
fi

echo "Created successful deployment backup: $final_backup"
echo "Only the current successful backup is retained."
