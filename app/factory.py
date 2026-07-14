from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Annotated, Any

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app import __version__
from app.graph import ChatAgent, encode_sse
from app.logging_config import configure_logging, log_kv
from app.observability.metrics import metrics
from app.schemas.chat import ChatRequest, ChatResponse, load_chat_openapi_examples
from app.services.inventory import build_inventory_payload
from app.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def create_app(*, settings: Settings | None = None, chat_agent: ChatAgent | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    configure_logging(app_settings.log_level)
    agent = chat_agent or ChatAgent(app_settings)
    chat_openapi_examples = load_chat_openapi_examples(
        "chat.json",
        summary="Complete chat request",
        description="Default values for the non-streaming chat request.",
    )
    chat_stream_openapi_examples = load_chat_openapi_examples(
        "chat-stream.json",
        summary="Complete streaming chat request",
        description="Default values for the streaming chat request.",
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        log_kv(
            logger,
            logging.INFO,
            "app_start",
            version=__version__,
            environment=app_settings.environment,
            backend=app_settings.llm_backend,
            ollama=app_settings.ollama_base_url,
            mcp_enabled=app_settings.mcp_enabled,
            mcp_url=app_settings.mcp_server_url,
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
                error=_safe_error(exc, expose=app_settings.expose_internal_health_details),
            )
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
            "Ollama model routing, and MCP tools."
        ),
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.state.chat_agent = agent
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
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
        return {"status": "ok", "service": current_settings.app_name, "version": __version__}

    @app.get("/health/ready")
    async def ready_health(request: Request) -> JSONResponse:
        current_agent: ChatAgent = request.app.state.chat_agent
        current_settings: Settings = request.app.state.settings
        payload: dict[str, Any] = {
            "status": "ready",
            "service": current_settings.app_name,
            "version": __version__,
            "dependencies": {},
        }
        ready = True
        if current_settings.llm_backend == "ollama":
            try:
                payload["dependencies"]["ollama"] = await current_agent.ollama.health()
            except Exception as exc:
                ready = False
                payload["dependencies"]["ollama"] = {
                    "status": "unavailable",
                    "error": _safe_error(
                        exc,
                        expose=current_settings.expose_internal_health_details,
                    ),
                }
        if current_settings.mcp_enabled:
            try:
                result = await current_agent.mcp.health_check()
                payload["dependencies"]["mcp"] = {
                    "status": "available" if result.ok else "unavailable",
                    "ok": result.ok,
                    "error": result.error,
                }
                ready = ready and result.ok
            except Exception as exc:
                ready = False
                payload["dependencies"]["mcp"] = {
                    "status": "unavailable",
                    "error": _safe_error(
                        exc,
                        expose=current_settings.expose_internal_health_details,
                    ),
                }
        if not ready:
            payload["status"] = "not_ready"
        return JSONResponse(
            status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload,
        )

    @app.get("/health/live")
    async def live_health(request: Request) -> JSONResponse:
        return await ready_health(request)

    @app.get("/api/inventory", dependencies=[Depends(require_api_key)])
    async def inventory(request: Request) -> dict[str, object]:
        current_agent: ChatAgent = request.app.state.chat_agent
        current_settings: Settings = request.app.state.settings
        live_inventory = await current_agent.load_inventory()
        return build_inventory_payload(current_settings, live_inventory, current_agent.selector)

    @app.get("/api/metrics", dependencies=[Depends(require_api_key)])
    async def application_metrics() -> dict[str, Any]:
        return {"service": app_settings.app_name, "version": __version__, **metrics.snapshot()}

    @app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
    async def chat(
        request: Request,
        chat_request: ChatRequest = Body(
            openapi_examples=deepcopy(chat_openapi_examples)
        ),
    ) -> ChatResponse:
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
        chat_request: ChatRequest = Body(
            openapi_examples=deepcopy(chat_stream_openapi_examples)
        ),
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
                    {"message": _safe_error(
                        exc,
                        expose=app_settings.expose_internal_health_details,
                    )},
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
