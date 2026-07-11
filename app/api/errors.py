from __future__ import annotations
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
logger=logging.getLogger(__name__)

def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(Exception)
    async def unhandled(_request: Request, exc: Exception):
        logger.exception('unhandled_request_error')
        return JSONResponse(status_code=500, content={'detail':'Internal server error.'})
