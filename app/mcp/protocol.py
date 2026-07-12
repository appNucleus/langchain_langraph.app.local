from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from app.mcp.errors import MCPProtocolError, MCPResponseMismatchError

JSONRPCId = int | str | None


@dataclass(frozen=True)
class JSONRPCResponse:
    request_id: JSONRPCId
    result: Any = None
    error: Any = None


def request_envelope(
    request_id: int | str,
    method: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not method:
        raise ValueError("JSON-RPC method must not be empty.")
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params or {},
    }


def notification_envelope(
    method: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not method:
        raise ValueError("JSON-RPC method must not be empty.")
    return {"jsonrpc": "2.0", "method": method, "params": params or {}}



def response_envelope(request_id: int | str, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def error_envelope(
    request_id: int | str,
    *,
    code: int,
    message: str,
    data: Any = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}

def validate_response(payload: Any, *, expected_id: JSONRPCId) -> JSONRPCResponse:
    if not isinstance(payload, dict):
        raise MCPProtocolError("MCP response must be a JSON object.")
    if payload.get("jsonrpc") != "2.0":
        raise MCPProtocolError(
            "MCP response has an invalid or missing jsonrpc version."
        )
    if "method" in payload:
        raise MCPProtocolError("A JSON-RPC request/notification is not a response.")
    if "id" not in payload:
        raise MCPProtocolError("MCP response is missing its JSON-RPC id.")

    response_id = payload.get("id")
    if expected_id is not None and response_id != expected_id:
        raise MCPResponseMismatchError(
            f"MCP response id {response_id!r} did not match request id "
            f"{expected_id!r}."
        )

    has_result = "result" in payload
    has_error = "error" in payload and payload.get("error") is not None
    if has_result == has_error:
        raise MCPProtocolError(
            "MCP response must contain exactly one of result or error."
        )

    error = payload.get("error")
    if has_error and not isinstance(error, dict):
        raise MCPProtocolError("MCP JSON-RPC error must be a JSON object.")

    return JSONRPCResponse(
        request_id=response_id,
        result=payload.get("result"),
        error=error,
    )


def select_response(
    messages: Iterable[dict[str, Any]],
    *,
    expected_id: int | str,
) -> JSONRPCResponse:
    """Select the matching response from JSON or an MCP SSE message sequence.

    Streamable HTTP SSE responses may contain server notifications or requests
    before the response to the initiating POST. Those messages must not be
    mistaken for the response merely because they appeared last in the stream.
    """

    response_ids: list[JSONRPCId] = []
    for payload in messages:
        if not isinstance(payload, dict):
            continue
        if payload.get("jsonrpc") != "2.0":
            # If it otherwise resembles a response, fail rather than silently
            # ignoring a malformed envelope.
            if "id" in payload and ("result" in payload or "error" in payload):
                validate_response(payload, expected_id=expected_id)
            continue
        if "method" in payload:
            # Server request or notification. The HTTP request remains pending
            # until the matching result/error message is found.
            continue
        if "result" not in payload and "error" not in payload:
            continue

        response_ids.append(payload.get("id"))
        if payload.get("id") == expected_id:
            return validate_response(payload, expected_id=expected_id)

    if response_ids:
        raise MCPResponseMismatchError(
            f"MCP response ids {response_ids!r} did not include request id "
            f"{expected_id!r}."
        )
    raise MCPProtocolError(
        f"MCP stream contained no JSON-RPC response for request id {expected_id!r}."
    )
