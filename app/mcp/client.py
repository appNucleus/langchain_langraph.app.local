from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from time import monotonic
from typing import Any

import httpx

from app import __version__
from app.logging_config import log_kv
from app.mcp.errors import (
    MCPError,
    MCPHTTPStatusError,
    MCPProtocolError,
    MCPSessionExpiredError,
)
from app.mcp.protocol import (
    error_envelope,
    notification_envelope,
    request_envelope,
    response_envelope,
    select_response,
)
from app.mcp.result_parser import parse_tool_result
from app.mcp.session import MCPSessionManager
from app.mcp.transport import MCPHTTPTransport
from app.observability.metrics import metrics
from app.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPToolResult:
    tool: str
    ok: bool
    data: Any
    error: str | None = None


class MCPClientError(MCPError):
    """Backward-compatible public MCP client exception."""


class MCPClient:
    """Session-aware async JSON-RPC client for Streamable HTTP MCP."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        http_transport: MCPHTTPTransport | None = None,
    ) -> None:
        self.settings = settings
        self._transport = http_transport or MCPHTTPTransport(
            settings,
            transport=transport,
        )
        self._request_id = 0
        self._id_lock = asyncio.Lock()
        self._session = MCPSessionManager(
            settings.mcp_protocol_version,
            settings.mcp_client_name,
            __version__,
        )
        self._tools_cache: list[dict[str, Any]] | None = None
        self._tools_cached_at = 0.0
        self._tools_lock = asyncio.Lock()

    @property
    def session_id(self) -> str | None:
        return self._session.state.session_id

    async def start(self) -> None:
        if not self.settings.mcp_enabled:
            return
        await self._transport.start()
        if (
            self.settings.mcp_session_enabled
            and self.settings.mcp_initialize_on_startup
        ):
            await self.initialize()

    async def aclose(self) -> None:
        if self.session_id:
            await self._transport.delete(headers=self._headers())
        await self._transport.close()
        self._session.reset()
        self.invalidate_tools_cache()

    async def initialize(self) -> None:
        if not self.settings.mcp_session_enabled:
            return
        await self._session.ensure_initialized(self._send_for_session)

    def invalidate_tools_cache(self) -> None:
        self._tools_cache = None
        self._tools_cached_at = 0.0

    async def list_tools(
        self,
        *,
        force_refresh: bool = False,
        allow_stale: bool = True,
    ) -> list[dict[str, Any]]:
        now = monotonic()
        if self._tools_fresh(now, force_refresh):
            metrics.inc("mcp.tools_cache_hit")
            return _copy_tools(self._tools_cache or [])

        async with self._tools_lock:
            now = monotonic()
            if self._tools_fresh(now, force_refresh):
                metrics.inc("mcp.tools_cache_hit")
                return _copy_tools(self._tools_cache or [])

            old_cache = self._tools_cache
            old_cached_at = self._tools_cached_at
            try:
                data = await self._rpc("tools/list", {}, retry_on_expiry=True)
                tools = (data or {}).get("tools") if isinstance(data, dict) else None
                if tools is None:
                    normalized: list[dict[str, Any]] = []
                elif not isinstance(tools, list):
                    raise MCPProtocolError(
                        "MCP tools/list result.tools must be a list."
                    )
                else:
                    normalized = [_validate_tool_definition(item) for item in tools]
            except Exception:
                if (
                    allow_stale
                    and old_cache is not None
                    and now - old_cached_at
                    <= self.settings.inventory_stale_if_error_seconds
                ):
                    metrics.inc("mcp.tools_stale_cache_used")
                    return _copy_tools(old_cache)
                raise

            self._tools_cache = normalized
            self._tools_cached_at = monotonic()
            metrics.inc("mcp.tools_refresh")
            return _copy_tools(normalized)

    async def health_check(self) -> MCPToolResult:
        return await self.call_tool("health_check", {})

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolResult:
        if not isinstance(name, str) or not name.strip():
            return MCPToolResult(
                tool=str(name),
                ok=False,
                data=None,
                error="MCP tool name must be a non-empty string.",
            )
        if arguments is not None and not isinstance(arguments, dict):
            return MCPToolResult(
                tool=name,
                ok=False,
                data=None,
                error="MCP tool arguments must be a JSON object.",
            )

        metrics.inc("mcp.tool_calls")
        try:
            # Never automatically replay tools/call after a session-expiration
            # response. The server may have executed the side effect before the
            # transport/session failure became visible to this client.
            result = await self._rpc(
                "tools/call",
                {"name": name, "arguments": arguments or {}},
                retry_on_expiry=False,
            )
            parsed = parse_tool_result(result)
            if not parsed.ok:
                metrics.inc("mcp.tool_reported_errors")
            return MCPToolResult(
                tool=name,
                ok=parsed.ok,
                data=parsed.data,
                error=parsed.error,
            )
        except Exception as exc:  # dependency failure becomes explicit tool data
            metrics.inc("mcp.tool_call_errors")
            error = _format_exception(exc)
            log_kv(
                logger,
                logging.ERROR,
                "mcp_client_tool_error",
                tool=name,
                error=error,
            )
            return MCPToolResult(tool=name, ok=False, data=None, error=error)

    async def _rpc(
        self,
        method: str,
        params: dict[str, Any],
        *,
        retry_on_expiry: bool,
    ) -> Any:
        if self.settings.mcp_session_enabled and method != "initialize":
            await self.initialize()

        observed_generation = self._session.generation
        try:
            result, _headers = await self._send_for_session(method, params, False)
            return result
        except MCPHTTPStatusError as exc:
            expired = (
                self.settings.mcp_session_enabled
                and self.session_id is not None
                and exc.status_code in {404, 410}
            )
            if not expired:
                raise

            metrics.inc("mcp.session_expired")
            log_kv(
                logger,
                logging.WARNING,
                "mcp_session_expired",
                status_code=exc.status_code,
                method=method,
            )
            self.invalidate_tools_cache()
            await self._session.recover_expired(
                self._send_for_session,
                observed_generation=observed_generation,
            )
            if retry_on_expiry:
                metrics.inc("mcp.session_safe_retries")
                result, _headers = await self._send_for_session(method, params, False)
                return result

            raise MCPSessionExpiredError(
                f"MCP session expired during {method}; the operation was not "
                "automatically replayed because its completion is ambiguous."
            ) from exc

    async def _send_for_session(
        self,
        method: str,
        params: dict[str, Any],
        notification: bool,
    ) -> tuple[Any, dict[str, str]]:
        request_id: int | None = None
        if notification:
            payload = notification_envelope(method, params)
        else:
            request_id = await self._next_id()
            payload = request_envelope(request_id, method, params)

        log_kv(
            logger,
            logging.DEBUG,
            "mcp_jsonrpc_request",
            method=method,
            session=bool(self.session_id),
        )
        messages, response_headers = await self._transport.post(
            payload,
            headers=self._headers(),
            allow_empty=notification,
        )
        normalized_headers = {
            key.lower(): value for key, value in response_headers.items()
        }
        if notification:
            return None, normalized_headers
        if not messages:
            raise MCPProtocolError(f"MCP method {method} returned an empty response.")

        for message in messages:
            method_name = message.get("method")
            if method_name == "notifications/tools/list_changed":
                self.invalidate_tools_cache()
                metrics.inc("mcp.tools_list_changed")
            if isinstance(method_name, str) and "id" in message:
                await self._respond_to_server_request(message)

        response = select_response(messages, expected_id=request_id)  # type: ignore[arg-type]
        if response.error is not None:
            raise MCPClientError(
                f"MCP JSON-RPC error for {method}: {response.error}"
            )
        return response.result, normalized_headers

    async def _respond_to_server_request(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        method = message.get("method")
        if not isinstance(request_id, (int, str)) or not isinstance(method, str):
            return

        if method == "ping":
            payload = response_envelope(request_id, {})
            metrics.inc("mcp.server_ping_requests")
        else:
            # The client declares no sampling, roots, or elicitation capabilities.
            # Return a JSON-RPC error instead of silently abandoning the server
            # request and leaving the MCP exchange incomplete.
            payload = error_envelope(
                request_id,
                code=-32601,
                message=f"Client does not support server method {method!r}.",
            )
            metrics.inc("mcp.unsupported_server_requests")

        await self._transport.post(
            payload,
            headers=self._headers(),
            allow_empty=True,
        )

    def _headers(self) -> dict[str, str]:
        protocol_version = (
            self._session.state.protocol_version
            or self.settings.mcp_protocol_version
        )
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": protocol_version,
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        return headers

    async def _next_id(self) -> int:
        async with self._id_lock:
            self._request_id += 1
            return self._request_id

    def _tools_fresh(self, now: float, force_refresh: bool) -> bool:
        return bool(
            not force_refresh
            and self._tools_cache is not None
            and now - self._tools_cached_at
            <= self.settings.inventory_cache_ttl_seconds
        )


def _validate_tool_definition(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MCPProtocolError("Each MCP tool definition must be a JSON object.")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise MCPProtocolError("Each MCP tool definition requires a non-empty name.")
    for key in ("inputSchema", "outputSchema"):
        schema = value.get(key)
        if schema is not None and not isinstance(schema, dict):
            raise MCPProtocolError(f"MCP tool {name!r} has an invalid {key}.")
    return dict(value)


def _copy_tools(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in items]


def _format_exception(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
