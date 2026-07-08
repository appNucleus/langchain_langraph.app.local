from __future__ import annotations

from collections.abc import AsyncIterator
import logging
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app import __version__
from app.graph import ChatAgent, encode_sse
from app.logging_config import configure_logging, log_kv
from app.services.inventory import build_inventory_payload
from app.schemas.chat import ChatRequest, ChatResponse
from app.settings import Settings, get_settings


def create_app(*, settings: Settings | None = None, chat_agent: ChatAgent | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    configure_logging(app_settings.log_level)
    logger = logging.getLogger(__name__)
    log_kv(
        logger,
        logging.INFO,
        "app_start",
        environment=app_settings.environment,
        backend=app_settings.llm_backend,
        ollama=app_settings.ollama_base_url,
        mcp_enabled=app_settings.mcp_enabled,
        mcp_url=app_settings.mcp_server_url,
        mcp_verify_tls=app_settings.mcp_verify_tls,
        mcp_follow_redirects=app_settings.mcp_follow_redirects,
    )
    agent = chat_agent or ChatAgent(app_settings)

    app = FastAPI(
        title=app_settings.app_name,
        version=__version__,
        description="FastAPI + LangGraph local assistant with Ollama model routing and MCP tools.",
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
            "health": "/health",
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
            "environment": current_settings.environment,
            "backend": current_settings.llm_backend,
            "log_level": current_settings.log_level,
            "api_key_enabled": bool(current_settings.api_key),
            "ollama_base_url": current_settings.ollama_base_url,
            "mcp_configured": bool(current_settings.mcp_server_url),
            "mcp_enabled": current_settings.mcp_enabled,
            "mcp_verify_tls": current_settings.mcp_verify_tls,
            "mcp_follow_redirects": current_settings.mcp_follow_redirects,
            "mcp_timeout_seconds": current_settings.mcp_timeout_seconds,
            "models": {
                "planner": current_settings.model_planner,
                "simple": current_settings.model_simple,
                "general": current_settings.model_general,
                "search": current_settings.model_search,
                "reasoning": current_settings.model_reasoning,
                "heavy": current_settings.model_heavy,
                "synthesis": current_settings.model_synthesis,
                "vision": current_settings.model_vision,
                "embedding": current_settings.embedding_model,
            },
        }

    @app.get("/api/inventory", dependencies=[Depends(require_api_key)])
    async def inventory(request: Request) -> dict[str, object]:
        """Return live Ollama models and live MCP tools available to this app."""
        agent: ChatAgent = request.app.state.chat_agent
        current_settings: Settings = request.app.state.settings
        live_inventory = await agent.load_inventory()
        return build_inventory_payload(current_settings, live_inventory, agent.selector)

    @app.get("/health/live")
    async def live_health(request: Request) -> dict[str, object]:
        """Optional live dependency health. Useful manually; not required for unit tests."""
        agent: ChatAgent = request.app.state.chat_agent
        current_settings: Settings = request.app.state.settings
        payload: dict[str, object] = {"status": "ok", "ollama": None, "mcp": None}
        if current_settings.llm_backend == "ollama":
            try:
                payload["ollama"] = await agent.ollama.health()
            except Exception as exc:  # noqa: BLE001
                payload["status"] = "degraded"
                payload["ollama"] = {"error": str(exc)}
        if current_settings.mcp_enabled:
            try:
                result = await agent.mcp.health_check()
                payload["mcp"] = {"ok": result.ok, "data": result.data, "error": result.error}
                if not result.ok:
                    payload["status"] = "degraded"
            except Exception as exc:  # noqa: BLE001
                payload["status"] = "degraded"
                payload["mcp"] = {"error": str(exc)}
        return payload

    @app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
    async def chat(request: Request, chat_request: ChatRequest) -> ChatResponse:
        agent: ChatAgent = request.app.state.chat_agent
        current_settings: Settings = request.app.state.settings
        log_kv(
            logging.getLogger(__name__),
            logging.INFO,
            "chat_request",
            thread_id=chat_request.thread_id,
            backend=current_settings.llm_backend,
            message_chars=len(chat_request.message),
        )
        return await agent.ainvoke(chat_request)

    @app.post("/api/chat/stream", dependencies=[Depends(require_api_key)])
    async def chat_stream(request: Request, chat_request: ChatRequest) -> StreamingResponse:
        agent: ChatAgent = request.app.state.chat_agent
        current_settings: Settings = request.app.state.settings
        log_kv(
            logging.getLogger(__name__),
            logging.INFO,
            "chat_stream_request",
            thread_id=chat_request.thread_id,
            backend=current_settings.llm_backend,
            message_chars=len(chat_request.message),
        )

        async def events() -> AsyncIterator[str]:
            try:
                async for item in agent.astream_events(chat_request):
                    yield encode_sse(item["event"], item["data"])
            except Exception as exc:  # noqa: BLE001
                yield encode_sse("error", {"message": str(exc)})

        return StreamingResponse(events(), media_type="text/event-stream")

    return app
