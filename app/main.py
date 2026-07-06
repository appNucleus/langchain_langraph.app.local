from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app import __version__
from app.graph import ChatAgent
from app.schemas.chat import ChatRequest, ChatResponse
from app.settings import Settings, get_settings

settings = get_settings()
chat_agent = ChatAgent(settings)

app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="Minimal FastAPI + LangGraph app server, ready to extend with Ollama, MCP, and databases.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    app_settings: Settings = Depends(get_settings),
) -> None:
    if app_settings.api_key and x_api_key != app_settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "version": __version__,
        "status": "running",
        "health": "/health",
        "chat": "/api/chat",
    }


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": __version__,
        "environment": settings.environment,
        "backend": settings.llm_backend,
        "ollama_model": settings.ollama_model if settings.llm_backend == "ollama" else None,
        "mcp_configured": bool(settings.mcp_server_url),
        "database_configured": bool(settings.database_url),
        "redis_configured": bool(settings.redis_url),
    }


@app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
async def chat(request: ChatRequest) -> ChatResponse:
    result = await chat_agent.graph.ainvoke(
        {
            "message": request.message,
            "system_prompt": request.system_prompt or settings.default_system_prompt,
            "metadata": request.metadata,
        }
    )
    return ChatResponse.from_result(
        thread_id=request.thread_id,
        response=result["response"],
        backend=result.get("backend", settings.llm_backend),
        model=result.get("model"),
        metadata=result.get("metadata", {}),
    )


@app.post("/api/chat/stream", dependencies=[Depends(require_api_key)])
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        yield "event: status\n"
        yield "data: {\"status\": \"started\"}\n\n"

        result = await chat_agent.graph.ainvoke(
            {
                "message": request.message,
                "system_prompt": request.system_prompt or settings.default_system_prompt,
                "metadata": request.metadata,
            }
        )

        # Minimal streaming placeholder. Later, replace this with native token streaming
        # from Ollama/LangGraph and tool-progress events.
        text = result["response"]
        for chunk in _chunk_text(text):
            payload = json.dumps({"delta": chunk}, ensure_ascii=False)
            yield "event: token\n"
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0)

        final_payload = json.dumps(
            {
                "thread_id": request.thread_id,
                "backend": result.get("backend", settings.llm_backend),
                "model": result.get("model"),
                "metadata": result.get("metadata", {}),
            },
            ensure_ascii=False,
        )
        yield "event: done\n"
        yield f"data: {final_payload}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def _chunk_text(text: str, size: int = 80) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)] or [""]
