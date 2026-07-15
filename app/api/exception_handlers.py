from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.observability.metrics import metrics
from app.orchestration.run_identity import RequestIdentityError

logger = logging.getLogger("app.factory")


async def unhandled_exception_handler(
    _request: Request,
    exc: Exception,
) -> JSONResponse:
    """Log unhandled failures while returning the existing generic response."""

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


def register_exception_handlers(app: FastAPI) -> None:
    """Register application exception handlers."""

    app.add_exception_handler(Exception, unhandled_exception_handler)


def request_error_to_http_exception(exc: RequestIdentityError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={"code": exc.error_code, "message": str(exc)},
    )


def safe_error(exc: BaseException, *, expose: bool) -> str:
    if expose:
        text = str(exc).strip()
        return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
    return "Dependency unavailable."
