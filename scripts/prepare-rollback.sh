#!/usr/bin/env bash
set -Eeuo pipefail

: "${BACKUP_ROOT:?BACKUP_ROOT is required}"
: "${DEPLOY_ENV_FILE:?DEPLOY_ENV_FILE is required}"
: "${ROLLBACK_STATE_DIR:?ROLLBACK_STATE_DIR is required}"
: "${CONTAINER_NAME:?CONTAINER_NAME is required}"
: "${ROLLBACK_IMAGE:?ROLLBACK_IMAGE is required}"

umask 077
rm -rf -- "$ROLLBACK_STATE_DIR"
mkdir -p "$ROLLBACK_STATE_DIR/source"
mkdir -p "$BACKUP_ROOT"

# Remove a stale rollback tag from an interrupted older run.
# Removing a tag does not remove an image that is still used by a container.
docker image rm "$ROLLBACK_IMAGE" >/dev/null 2>&1 || true

if ! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "No existing container was found."
  echo "This is a first deployment; automatic rollback is unavailable."
  printf 'false\n' > "$ROLLBACK_STATE_DIR/has-rollback"
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    echo "HAS_ROLLBACK=false" >> "$GITHUB_ENV"
  fi
  exit 0
fi

previous_image_id="$(docker container inspect --format='{{.Image}}' "$CONTAINER_NAME")"
docker image tag "$previous_image_id" "$ROLLBACK_IMAGE"
printf '%s\n' "$previous_image_id" > "$ROLLBACK_STATE_DIR/previous-image-id.txt"
printf 'true\n' > "$ROLLBACK_STATE_DIR/has-rollback"

# Prefer the latest backup created by this deployment system.
previous_backup=""
while IFS= read -r candidate; do
  if [[ -f "$candidate/.langchain-langraph-success-backup" && \
        -f "$candidate/source/compose.yaml" && \
        -f "$candidate/runtime.env" ]]; then
    previous_backup="$candidate"
    break
  fi
done < <(
  find "$BACKUP_ROOT" \
    -mindepth 1 \
    -maxdepth 1 \
    -type d \
    ! -name '.staging-*' \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | cut -d' ' -f2-
)

if [[ -n "$previous_backup" ]]; then
  cp -a "$previous_backup" "$ROLLBACK_STATE_DIR/previous-backup"
  cp -a "$previous_backup/source/." "$ROLLBACK_STATE_DIR/source/"
  install -m 600 "$previous_backup/runtime.env" "$ROLLBACK_STATE_DIR/runtime.env"
  printf '%s\n' "$previous_backup" > "$ROLLBACK_STATE_DIR/previous-backup-path.txt"
  echo "Prepared rollback from managed backup: $previous_backup"
else
  git archive --format=tar HEAD | tar -xf - -C "$ROLLBACK_STATE_DIR/source"
  install -m 600 "$DEPLOY_ENV_FILE" "$ROLLBACK_STATE_DIR/runtime.env"
  echo "No previous configuration snapshot exists; using the current Compose configuration as rollback source fallback."
fi

if [[ -n "${GITHUB_ENV:-}" ]]; then
  echo "HAS_ROLLBACK=true" >> "$GITHUB_ENV"
fi

echo "Rollback image: $ROLLBACK_IMAGE"
echo "Rollback image ID: $previous_image_id"
