from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.exception_handlers import safe_error, unhandled_exception_handler
from app.api.routes.health import ready_health
from app.factory import create_app
from app.graph import ChatAgent
from app.orchestration.run_identity import RequestIdentityError
from app.settings import Settings


EXPECTED_API_OPERATIONS = [
    ("GET", "/"),
    ("GET", "/health"),
    ("GET", "/health/live"),
    ("GET", "/health/ready"),
    ("GET", "/api/inventory"),
    ("GET", "/api/metrics"),
    ("POST", "/api/chat"),
    ("POST", "/api/chat/stream"),
]
OPENAPI_SCHEMA_PATH = "/openapi" + "." + "json"


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_backend="echo",
        mcp_enabled=False,
        ollama_required=False,
        mcp_required=False,
        persistence_required=False,
        artifact_storage_required=False,
        final_verification_enabled=False,
        state_backend="memory",
        checkpoint_backend="memory",
        artifact_backend="disabled",
    )


def _app_operations() -> list[tuple[str, str]]:
    settings = _settings()
    app = create_app(settings=settings, chat_agent=ChatAgent(settings))
    # FastAPI 0.139 keeps included routers nested; OpenAPI is the stable public contract.
    schema = app.openapi()
    return [
        (method.upper(), path)
        for path, path_item in schema["paths"].items()
        for method in path_item
        if method in {"get", "post"}
    ]


@contextmanager
def _client() -> Iterator[TestClient]:
    settings = _settings()
    app = create_app(settings=settings, chat_agent=ChatAgent(settings))
    with TestClient(app) as client:
        yield client


def test_public_api_method_path_combinations_are_unchanged() -> None:
    assert _app_operations() == EXPECTED_API_OPERATIONS


def test_public_api_method_path_combinations_are_unique() -> None:
    operations = _app_operations()
    assert len(operations) == len(set(operations))


def test_root_metrics_and_openapi_remain_registered() -> None:
    with _client() as client:
        root = client.get("/")
        inventory = client.get("/api/inventory")
        metrics = client.get("/api/metrics")
        openapi = client.get(OPENAPI_SCHEMA_PATH)

    assert root.status_code == 200
    assert root.json()["chat"] == "/api/chat"
    assert inventory.status_code == 200
    inventory_payload = inventory.json()
    assert "models" in inventory_payload["ollama"]
    assert "tools" in inventory_payload["mcp"]
    assert metrics.status_code == 200
    assert "service" in metrics.json()

    schema = openapi.json()
    operations = {
        (method.upper(), path)
        for path, path_item in schema["paths"].items()
        for method in path_item
        if method in {"get", "post"}
    }
    assert operations == set(EXPECTED_API_OPERATIONS)
    for path in ("/api/chat", "/api/chat/stream"):
        examples = schema["paths"][path]["post"]["requestBody"]["content"][
            "application/json"
        ]["examples"]
        assert examples


@pytest.mark.asyncio
async def test_generic_error_helpers_preserve_external_error_behavior() -> None:
    response = await unhandled_exception_handler(None, RuntimeError("internal detail"))  # type: ignore[arg-type]

    assert response.status_code == 500
    assert response.body == b'{"detail":"Internal server error."}'
    assert safe_error(RuntimeError("internal detail"), expose=False) == (
        "Dependency unavailable."
    )
    assert safe_error(RuntimeError("internal detail"), expose=True) == (
        "RuntimeError: internal detail"
    )


class _IdentityErrorAgent(ChatAgent):
    async def ainvoke(self, _request):
        raise RequestIdentityError("invalid request identity")

    async def astream_events(self, _request):
        if False:
            yield None
        raise RequestIdentityError("invalid request identity")


class _StreamFailureAgent(ChatAgent):
    async def astream_events(self, _request):
        if False:
            yield None
        raise RuntimeError("private stream failure")


def test_request_identity_errors_keep_http_and_sse_contracts() -> None:
    settings = _settings()
    app = create_app(settings=settings, chat_agent=_IdentityErrorAgent(settings))

    with TestClient(app) as client:
        chat = client.post("/api/chat", json={"message": "hello"})
        stream = client.post("/api/chat/stream", json={"message": "hello"})

    assert chat.status_code == 400
    assert chat.json()["detail"]["code"] == "request_identity_error"
    assert "event: error" in stream.text
    assert "request_identity_error" in stream.text


def test_stream_unhandled_errors_remain_generic() -> None:
    settings = _settings()
    app = create_app(settings=settings, chat_agent=_StreamFailureAgent(settings))

    with TestClient(app) as client:
        stream = client.post("/api/chat/stream", json={"message": "hello"})

    assert stream.status_code == 200
    assert "event: error" in stream.text
    assert "Dependency unavailable." in stream.text
    assert "private stream failure" not in stream.text


class _InventorySnapshot:
    def __init__(self, *, errors, model_names, tool_names, cached=False):
        self.errors = errors
        self.model_names = model_names
        self.tool_names = tool_names
        self.cached = cached


class _InventoryService:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    async def load(self):
        return self.snapshot


class _ReadyAgent:
    def __init__(self, persistence=None, failure=None):
        self.persistence = persistence
        self.failure = failure

    async def persistence_health(self):
        if self.failure is not None:
            raise self.failure
        return self.persistence

    def dependency_startup_status(self):
        return {"startup": {"status": "available"}}


@pytest.mark.asyncio
async def test_readiness_preserves_required_dependency_failure_behavior() -> None:
    settings = _settings()
    settings.llm_backend = "ollama"
    settings.mcp_enabled = True
    settings.ollama_required = True
    settings.mcp_required = True
    settings.persistence_required = True
    settings.artifact_storage_required = True
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                settings=settings,
                chat_agent=_ReadyAgent(failure=RuntimeError("private persistence detail")),
                inventory_service=_InventoryService(
                    _InventorySnapshot(
                        errors={"ollama": "down", "mcp": "down"},
                        model_names=[],
                        tool_names=[],
                    )
                ),
            )
        )
    )

    response = await ready_health(request)  # type: ignore[arg-type]

    assert response.status_code == 503
    assert b'"status":"not_ready"' in response.body
    assert b"private persistence detail" not in response.body
    assert b"Dependency unavailable." in response.body


@pytest.mark.asyncio
async def test_readiness_preserves_available_dependency_behavior() -> None:
    settings = _settings()
    settings.llm_backend = "ollama"
    settings.mcp_enabled = True
    settings.ollama_required = True
    settings.mcp_required = True
    settings.persistence_required = True
    settings.artifact_storage_required = True
    persistence = {
        "conversation": {"status": "available"},
        "runs": {"status": "available"},
        "checkpoint": {"status": "available"},
        "artifacts": {"status": "disabled"},
    }
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                settings=settings,
                chat_agent=_ReadyAgent(persistence=persistence),
                inventory_service=_InventoryService(
                    _InventorySnapshot(
                        errors={},
                        model_names=["model"],
                        tool_names=["tool"],
                        cached=True,
                    )
                ),
            )
        )
    )

    response = await ready_health(request)  # type: ignore[arg-type]

    assert response.status_code == 200
    assert b'"status":"ready"' in response.body
    assert b'"cached":true' in response.body
