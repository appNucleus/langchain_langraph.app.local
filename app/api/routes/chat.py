from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.dependencies import require_api_key
from app.api.exception_handlers import request_error_to_http_exception, safe_error
from app.graph import ChatAgent, encode_sse
from app.observability.metrics import metrics
from app.orchestration.run_identity import RequestIdentityError
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger("app.factory")
router = APIRouter(tags=["Chat"])


@router.post(
    "/api/chat",
    response_model=ChatResponse,
    summary="Send a chat message",
    dependencies=[Depends(require_api_key)],
)
async def chat(request: Request, chat_request: ChatRequest) -> ChatResponse:
    current_agent: ChatAgent = request.app.state.chat_agent
    metrics.inc("api.chat.requests")
    try:
        return await current_agent.ainvoke(chat_request)
    except RequestIdentityError as exc:
        metrics.inc(f"api.chat.{exc.error_code}")
        raise request_error_to_http_exception(exc) from exc
    except asyncio.CancelledError:
        metrics.inc("api.chat.cancelled")
        raise


@router.post("/api/chat/stream", summary="Stream Chat Response", dependencies=[Depends(require_api_key)])
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
                {"message": safe_error(exc, expose=False)},
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
