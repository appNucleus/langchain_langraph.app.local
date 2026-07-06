# Minimal Makefile for local FastAPI + LangGraph Docker Compose development.

.DEFAULT_GOAL := help

COMPOSE := docker compose
ENV_FILE := .env
SERVICE := app
CONTAINER := langchain-langraph-app
HOST_BIND := 127.0.0.1
APP_PORT := 8001
LOCAL_API_URL := http://$(HOST_BIND):$(APP_PORT)

.PHONY: help env ensure-env doctor config build run up debug stop down restart rebuild logs ps status health smoke test syntax shell sh clean clean-dangling url

help:
	@echo
	@echo "LangChain LangGraph app commands"
	@echo
	@echo "make env             Create/normalize .env for local Docker use"
	@echo "make run             Build and run in background on 127.0.0.1:8001"
	@echo "make debug           Build and run in foreground with live logs"
	@echo "make stop            Stop containers without deleting them"
	@echo "make down            Stop and remove containers/networks, keep volumes"
	@echo "make restart         Stop, then run again"
	@echo "make rebuild         Rebuild without cache, then run"
	@echo "make logs            Follow Docker Compose logs"
	@echo "make ps              Show Compose container status"
	@echo "make health          Check Docker health and HTTP health endpoint"
	@echo "make smoke           Run a real HTTP smoke test"
	@echo "make test            Run pytest inside a temporary Python Docker container"
	@echo "make syntax          Python syntax compile check inside Docker"
	@echo "make shell           Open sh inside the running app container"
	@echo "make clean-dangling  Remove dangling Docker images only, not volumes"
	@echo "make url             Print local API base URL"
	@echo

env: ensure-env

ensure-env:
	@if [ ! -f "$(ENV_FILE)" ]; then cp .env.example "$(ENV_FILE)"; fi
	@python - <<'PY'
from pathlib import Path
path = Path('.env')
lines = path.read_text(encoding='utf-8').splitlines()
values = {'HOST_BIND': '127.0.0.1', 'APP_PORT': '8001'}
seen = set()
out = []
for line in lines:
    if '=' in line and not line.lstrip().startswith('#'):
        key = line.split('=', 1)[0]
        if key in values:
            out.append(f'{key}={values[key]}')
            seen.add(key)
            continue
    out.append(line)
for key, value in values.items():
    if key not in seen:
        out.append(f'{key}={value}')
path.write_text('\n'.join(out) + '\n', encoding='utf-8')
print('.env is ready: HOST_BIND=127.0.0.1, APP_PORT=8001')
PY

doctor:
	docker version
	docker compose version

config: ensure-env
	$(COMPOSE) --env-file $(ENV_FILE) config

build: ensure-env
	$(COMPOSE) --env-file $(ENV_FILE) build --pull

run: ensure-env
	$(COMPOSE) --env-file $(ENV_FILE) up --build --detach --remove-orphans
	@echo
	@echo "Running locally at $(LOCAL_API_URL)"
	@echo "Use 'make smoke' to verify the API."

up: run

debug: ensure-env
	$(COMPOSE) --env-file $(ENV_FILE) up --build --remove-orphans

stop: ensure-env
	$(COMPOSE) --env-file $(ENV_FILE) stop

down: ensure-env
	$(COMPOSE) --env-file $(ENV_FILE) down --remove-orphans

restart: stop run

rebuild: ensure-env
	$(COMPOSE) --env-file $(ENV_FILE) build --pull --no-cache
	$(COMPOSE) --env-file $(ENV_FILE) up --detach --remove-orphans
	@echo
	@echo "Rebuilt and running locally at $(LOCAL_API_URL)"

logs: ensure-env
	$(COMPOSE) --env-file $(ENV_FILE) logs --tail=150 --follow

ps: ensure-env
	$(COMPOSE) --env-file $(ENV_FILE) ps

status: ps

health:
	docker inspect --format="{{.State.Health.Status}}" $(CONTAINER)
	curl --fail --silent --show-error $(LOCAL_API_URL)/health
	@echo

smoke:
	curl --fail --silent --show-error $(LOCAL_API_URL)/health
	@echo
	curl --fail --silent --show-error \
	  -H 'Content-Type: application/json' \
	  -d '{"message":"hello"}' \
	  $(LOCAL_API_URL)/api/chat
	@echo

test:
	docker run --rm -v "$(CURDIR):/workspace" -w /workspace python:3.12-slim sh -c "pip install --no-cache-dir -r requirements-dev.txt && python -m pytest"

syntax:
	docker run --rm -v "$(CURDIR):/workspace" -w /workspace python:3.12-slim python -m compileall app tests

shell:
	docker exec -it $(CONTAINER) sh

sh: shell

clean: down

clean-dangling:
	docker image prune --force

url:
	@echo $(LOCAL_API_URL)
