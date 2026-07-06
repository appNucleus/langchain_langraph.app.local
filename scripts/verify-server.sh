#!/usr/bin/env bash
set -Eeuo pipefail

for command_name in docker git curl; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "Missing required command: $command_name" >&2
    exit 1
  }
done

docker version >/dev/null
docker compose version >/dev/null
docker ps >/dev/null

echo "Docker Engine: OK"
echo "Docker Compose: OK"
echo "Docker access: OK"
echo "Git: OK"
echo "curl: OK"
