from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app import __version__
from app.graph import ChatAgent, encode_sse
from app.logging_config import configure_logging, log_kv
from app.observability.metrics import metrics
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.inventory import InventoryService, build_inventory_payload
from app.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def create_app(
    *,
    settings: Settings | None = None,
    chat_agent: ChatAgent | None = None,
) -> FastAPI:
    app_settings = settings or get_settings()
    configure_logging(app_settings.log_level)
    agent = chat_agent or ChatAgent(app_settings)
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
            checkpoint_backend=app_settings.checkpoint_backend,
            artifact_backend=app_settings.artifact_backend,
        )
        try:
            start = getattr(agent, "start", None)
            if callable(start):
                await start()
        except Exception as exc:
            metrics.inc("app.startup_dependency_error")
            log_kv(
                logger,
                logging.ERROR,
                "app_dependency_start_error",
                error=_safe_error(
                    exc,
                    expose=app_settings.expose_internal_health_details,
                ),
            )
            # Preserve the existing Phase 3/4 startup policy. Phase 1 only
            # activates shared clients and inventory caching.
            if app_settings.persistence_required:
                raise
        try:
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
            "FastAPI + LangGraph local assistant with bounded agent execution, "
            "durable checkpoints, Ollama model routing, and MCP tools."
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
        log_kv(logger, logging.ERROR, "api_unhandled_error", error=repr(exc))
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

    @app.get("/health/ready")
    async def ready_health(request: Request) -> JSONResponse:
        current_agent: ChatAgent = request.app.state.chat_agent
        current_settings: Settings = request.app.state.settings
        current_inventory_service: InventoryService = request.app.state.inventory_service
        payload: dict[str, Any] = {
            "status": "ready",
            "service": current_settings.app_name,
            "version": __version__,
            "dependencies": {},
        }
        ready = True

        # Reuse the single-flight TTL inventory rather than forcing /api/tags and
        # MCP health tool calls on every orchestrator probe. The first probe still
        # validates both dependencies; later probes are cheap until the TTL expires.
        live_inventory = await current_inventory_service.load()
        if current_settings.llm_backend == "ollama":
            ollama_error = live_inventory.errors.get("ollama")
            ollama_ready = not ollama_error and bool(live_inventory.model_names)
            payload["dependencies"]["ollama"] = {
                "status": "available" if ollama_ready else "unavailable",
                "model_count": len(live_inventory.model_names),
                "cached": live_inventory.cached,
                "error": (
                    ollama_error
                    if current_settings.expose_internal_health_details
                    else ("Dependency unavailable." if ollama_error else None)
                ),
            }
            ready = ready and ollama_ready

        if current_settings.mcp_enabled:
            mcp_error = live_inventory.errors.get("mcp")
            mcp_ready = not mcp_error
            payload["dependencies"]["mcp"] = {
                "status": "available" if mcp_ready else "unavailable",
                "tool_count": len(live_inventory.tool_names),
                "cached": live_inventory.cached,
                "error": (
                    mcp_error
                    if current_settings.expose_internal_health_details
                    else ("Dependency unavailable." if mcp_error else None)
                ),
            }
            ready = ready and mcp_ready

        try:
            payload["dependencies"]["persistence"] = (
                await current_agent.persistence_health()
            )
        except Exception as exc:
            if current_settings.persistence_enabled:
                ready = False
            payload["dependencies"]["persistence"] = {
                "status": "unavailable",
                "error": _safe_error(
                    exc,
                    expose=current_settings.expose_internal_health_details,
                ),
            }

        if not ready:
            payload["status"] = "not_ready"
        return JSONResponse(
            status_code=(
                status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            content=payload,
        )

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

    @app.get("/api/inventory", dependencies=[Depends(require_api_key)])
    async def inventory(request: Request) -> dict[str, object]:
        current_agent: ChatAgent = request.app.state.chat_agent
        current_settings: Settings = request.app.state.settings
        current_inventory_service: InventoryService = (
            request.app.state.inventory_service
        )
        live_inventory = await current_inventory_service.load()
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
            except asyncio.CancelledError:
                metrics.inc("api.chat_stream.cancelled")
                raise
            except Exception as exc:
                metrics.inc("api.chat_stream.error")
                yield encode_sse(
                    "error",
                    {
                        "message": _safe_error(
                            exc,
                            expose=app_settings.expose_internal_health_details,
                        )
                    },
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

    return app


def _safe_error(exc: BaseException, *, expose: bool) -> str:
    if expose:
        text = str(exc).strip()
        return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
    return "Dependency unavailable."
