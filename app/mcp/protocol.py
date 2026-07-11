from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.mcp.errors import MCPProtocolError, MCPResponseMismatchError


@dataclass(frozen=True)
class JSONRPCResponse:
    request_id: int | str | None
    result: Any = None
    error: Any = None


def request_envelope(request_id: int, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}


def notification_envelope(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "method": method, "params": params or {}}


def validate_response(payload: Any, *, expected_id: int | str | None) -> JSONRPCResponse:
    if not isinstance(payload, dict):
        raise MCPProtocolError("MCP response must be a JSON object.")
    if payload.get("jsonrpc") != "2.0":
        raise MCPProtocolError("MCP response has an invalid or missing jsonrpc version.")
    response_id = payload.get("id")
    if expected_id is not None and response_id != expected_id:
        raise MCPResponseMismatchError(
            f"MCP response id {response_id!r} did not match request id {expected_id!r}."
        )
    has_result = "result" in payload
    has_error = "error" in payload and payload.get("error") is not None
    if has_result == has_error:
        raise MCPProtocolError("MCP response must contain exactly one of result or error.")
    return JSONRPCResponse(request_id=response_id, result=payload.get("result"), error=payload.get("error"))
