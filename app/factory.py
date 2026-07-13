from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app import __version__
from app.graph import ChatAgent, encode_sse
from app.logging_config import configure_logging, log_kv
from app.observability.metrics import metrics
from app.orchestration.chat_runtime import ChatRuntimeAgent
from app.orchestration.run_identity import RequestIdentityError
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    build_chat_request_openapi_examples,
    load_chat_request_example,
)
from app.services.inventory import InventoryService, build_inventory_payload
from app.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def create_app(
    *,
    settings: Settings | None = None,
    chat_agent: ChatAgent | None = None,
) -> FastAPI:
    app_settings = settings or get_settings()
    chat_request_openapi_examples = build_chat_request_openapi_examples(
        load_chat_request_example()
    )
    configure_logging(app_settings.log_level)
    agent = chat_agent or ChatRuntimeAgent(app_settings)
    inventory_service = getattr(agent, "inventory_service", None)
    if inventory_service is None:
        inventory_service = InventoryService(app_settings, agent.ollama, agent.mcp)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        log_kv(
            logger,
            logging.INFO,
            "app_start",
            version=__version__,
            environment=app_settings.environment,
            backend=app_settings.llm_backend,
            state_backend=app_settings.state_backend,
            run_repository_backend=app_settings.run_repository_backend,
            checkpoint_backend=app_settings.checkpoint_backend,
            artifact_backend=app_settings.artifact_backend,
        )
        try:
            start = getattr(agent, "start", None)
            if callable(start):
                await start()
            yield
        finally:
            close = getattr(agent, "aclose", None)
            if callable(close):
                await close()
            log_kv(logger, logging.INFO, "app_stop", version=__version__)

    app = FastAPI(
        title=app_settings.app_name,
        version=__version__,
        description=(
            "FastAPI + LangGraph local assistant with bounded execution, "
            "durable run outcomes, checkpoints, Ollama model routing, and MCP tools."
        ),
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.state.chat_agent = agent
    app.state.inventory_service = inventory_service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        _request: Request,
        exc: Exception,
    ) -> JSONResponse:
        metrics.inc("api.unhandled_error")
        logger.error(
            "api_unhandled_error error=%r",
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error."},
        )

    async def require_api_key(
        request: Request,
        x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ) -> None:
        current_settings: Settings = request.app.state.settings
        if current_settings.api_key and x_api_key != current_settings.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key.",
            )

    @app.get("/")
    async def root(request: Request) -> dict[str, str]:
        current_settings: Settings = request.app.state.settings
        return {
            "service": current_settings.app_name,
            "version": __version__,
            "status": "running",
            "liveness": "/health",
            "readiness": "/health/ready",
            "chat": "/api/chat",
            "stream": "/api/chat/stream",
            "inventory": "/api/inventory",
        }

    @app.get("/health")
    async def health(request: Request) -> dict[str, object]:
        current_settings: Settings = request.app.state.settings
        return {
            "status": "ok",
            "service": current_settings.app_name,
            "version": __version__,
        }

    @app.get("/health/live")
    async def live_health(request: Request) -> JSONResponse:
        current_settings: Settings = request.app.state.settings
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "alive",
                "service": current_settings.app_name,
                "version": __version__,
            },
        )

    @app.get("/health/ready")
    async def ready_health(request: Request) -> JSONResponse:
        current_agent: ChatAgent = request.app.state.chat_agent
        current_settings: Settings = request.app.state.settings
        service: InventoryService = request.app.state.inventory_service
        payload: dict[str, Any] = {
            "status": "ready",
            "service": current_settings.app_name,
            "version": __version__,
            "dependencies": {},
        }
        ready = True
        live_inventory = await service.load()

        if current_settings.llm_backend == "ollama":
            error = live_inventory.errors.get("ollama")
            available = not error and bool(live_inventory.model_names)
            payload["dependencies"]["ollama"] = {
                "status": "available" if available else "unavailable",
                "model_count": len(live_inventory.model_names),
                "cached": live_inventory.cached,
                "error": (
                    error
                    if current_settings.expose_internal_health_details
                    else ("Dependency unavailable." if error else None)
                ),
            }
            if current_settings.ollama_required:
                ready = ready and available

        if current_settings.mcp_enabled:
            error = live_inventory.errors.get("mcp")
            available = not error
            payload["dependencies"]["mcp"] = {
                "status": "available" if available else "unavailable",
                "tool_count": len(live_inventory.tool_names),
                "cached": live_inventory.cached,
                "error": (
                    error
                    if current_settings.expose_internal_health_details
                    else ("Dependency unavailable." if error else None)
                ),
            }
            if current_settings.mcp_required:
                ready = ready and available

        try:
            persistence = await current_agent.persistence_health()
            payload["dependencies"]["persistence"] = persistence
            conversation_ok = (
                (persistence.get("conversation") or {}).get("status") == "available"
            )
            runs_ok = (persistence.get("runs") or {}).get("status") == "available"
            checkpoint_ok = (
                (persistence.get("checkpoint") or {}).get("status") == "available"
            )
            artifacts_ok = (persistence.get("artifacts") or {}).get("status") in {
                "available",
                "disabled",
            }
            if current_settings.persistence_required:
                ready = ready and conversation_ok and runs_ok and checkpoint_ok
            if current_settings.artifact_storage_required:
                ready = ready and artifacts_ok
        except Exception as exc:
            if (
                current_settings.persistence_required
                or current_settings.artifact_storage_required
            ):
                ready = False
            payload["dependencies"]["persistence"] = {
                "status": "unavailable",
                "error": _safe_error(
                    exc,
                    expose=current_settings.expose_internal_health_details,
                ),
            }

        startup_status = getattr(current_agent, "dependency_startup_status", None)
        if callable(startup_status):
            payload["startup"] = startup_status()
        if not ready:
            payload["status"] = "not_ready"
        return JSONResponse(
            status_code=(
                status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            content=payload,
        )

    @app.get("/api/inventory", dependencies=[Depends(require_api_key)])
    async def inventory(request: Request) -> dict[str, object]:
        current_agent: ChatAgent = request.app.state.chat_agent
        current_settings: Settings = request.app.state.settings
        service: InventoryService = request.app.state.inventory_service
        live_inventory = await service.load()
        return build_inventory_payload(
            current_settings,
            live_inventory,
            current_agent.selector,
        )

    @app.get("/api/metrics", dependencies=[Depends(require_api_key)])
    async def application_metrics() -> dict[str, Any]:
        return {
            "service": app_settings.app_name,
            "version": __version__,
            **metrics.snapshot(),
        }

    @app.post(
        "/api/chat",
        response_model=ChatResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def chat(request: Request, chat_request: ChatRequest) -> ChatResponse:
        current_agent: ChatAgent = request.app.state.chat_agent
        metrics.inc("api.chat.requests")
        try:
            return await current_agent.ainvoke(chat_request)
        except RequestIdentityError as exc:
            metrics.inc(f"api.chat.{exc.error_code}")
            raise _request_error_to_http_exception(exc) from exc
        except asyncio.CancelledError:
            metrics.inc("api.chat.cancelled")
            raise

    @app.post("/api/chat/stream", dependencies=[Depends(require_api_key)])
    async def chat_stream(
        request: Request,
        chat_request: ChatRequest,
    ) -> StreamingResponse:
        current_agent: ChatAgent = request.app.state.chat_agent
        metrics.inc("api.chat_stream.requests")

        async def events() -> AsyncIterator[str]:
            try:
                async for item in current_agent.astream_events(chat_request):
                    if await request.is_disconnected():
                        metrics.inc("api.chat_stream.disconnected")
                        break
                    yield encode_sse(str(item["event"]), item["data"])
            except RequestIdentityError as exc:
                metrics.inc(f"api.chat_stream.{exc.error_code}")
                yield encode_sse(
                    "error",
                    {
                        "code": exc.error_code,
                        "status_code": exc.status_code,
                        "message": str(exc),
                    },
                )
            except asyncio.CancelledError:
                metrics.inc("api.chat_stream.cancelled")
                raise
            except Exception as exc:
                metrics.inc("api.chat_stream.error")
                logger.exception("chat_stream_error")
                yield encode_sse(
                    "error",
                    {"message": _safe_error(exc, expose=False)},
                )

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    generated_openapi = app.openapi

    def openapi_with_chat_request_example() -> dict[str, Any]:
        schema = generated_openapi()
        for path in ("/api/chat", "/api/chat/stream"):
            operation = (schema.get("paths") or {}).get(path, {}).get("post", {})
            content = (operation.get("requestBody") or {}).get("content", {})
            json_body = content.get("application/json")
            if isinstance(json_body, dict):
                json_body["examples"] = deepcopy(chat_request_openapi_examples)
        return schema

    app.openapi = openapi_with_chat_request_example  # type: ignore[method-assign]
    return app


def _request_error_to_http_exception(exc: RequestIdentityError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={"code": exc.error_code, "message": str(exc)},
    )


def _safe_error(exc: BaseException, *, expose: bool) -> str:
    if expose:
        text = str(exc).strip()
        return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
    return "Dependency unavailable."
