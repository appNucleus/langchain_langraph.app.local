from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app import __version__
from app.api.errors import install_exception_handlers
from app.api.streaming import encode_sse
from app.graph import ChatAgent
from app.observability.metrics import metrics
from app.schemas.chat import ChatRequest, ChatResponse
from app.settings import Settings, get_settings


def create_app(*, settings: Settings | None = None, chat_agent: ChatAgent | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    agent = chat_agent or ChatAgent(app_settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await agent.start()
        try:
            yield
        finally:
            await agent.aclose()

    app = FastAPI(title=app_settings.app_name, version=__version__, lifespan=lifespan)
    app.state.settings = app_settings
    app.state.chat_agent = agent
    app.add_middleware(CORSMiddleware, allow_origins=app_settings.cors_origins, allow_credentials=False,
                       allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])
    install_exception_handlers(app)

    async def require_api_key(request: Request, x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None) -> None:
        current: Settings = request.app.state.settings
        if current.api_key and x_api_key != current.api_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key.")

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"service": app_settings.app_name, "version": __version__, "status": "running"}

    @app.get("/health")
    async def liveness() -> dict[str, object]:
        return {"status": "ok", "service": app_settings.app_name, "version": __version__}

    @app.get("/health/ready")
    async def readiness(request: Request):
        current: Settings = request.app.state.settings
        current_agent: ChatAgent = request.app.state.chat_agent
        payload: dict[str, object] = {"status": "ready", "dependencies": {}}
        if current.llm_backend == "ollama":
            try: payload["dependencies"]["ollama"] = await current_agent.ollama.health()
            except Exception: payload["status"] = "not_ready"; payload["dependencies"]["ollama"] = {"status": "unavailable"}
        if current.mcp_enabled:
            try:
                result = await current_agent.mcp.health_check()
                payload["dependencies"]["mcp"] = {"ok": result.ok}
                if not result.ok: payload["status"] = "not_ready"
            except Exception: payload["status"] = "not_ready"; payload["dependencies"]["mcp"] = {"status": "unavailable"}
        if payload["status"] != "ready":
            raise HTTPException(status_code=503, detail=payload)
        return payload

    @app.get("/health/live")
    async def legacy_live(request: Request):
        return await readiness(request)

    @app.get("/api/inventory", dependencies=[Depends(require_api_key)])
    async def inventory(request: Request):
        return await request.app.state.chat_agent.load_inventory()

    @app.get("/api/metrics", dependencies=[Depends(require_api_key)])
    async def app_metrics():
        return metrics.snapshot()

    @app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
    async def chat(request: Request, chat_request: ChatRequest) -> ChatResponse:
        return await request.app.state.chat_agent.ainvoke(chat_request)

    @app.post("/api/chat/stream", dependencies=[Depends(require_api_key)])
    async def chat_stream(request: Request, chat_request: ChatRequest) -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            async for item in request.app.state.chat_agent.astream_events(chat_request):
                if await request.is_disconnected():
                    break
                yield encode_sse(str(item["event"]), item)
        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    return app
