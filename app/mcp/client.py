from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from app.settings import Settings


@dataclass(frozen=True)
class MCPToolResult:
    tool: str
    ok: bool
    data: Any
    error: str | None = None


class MCPClientError(RuntimeError):
    pass


class MCPClient:
    """Async JSON-RPC client for the local MCP HTTP endpoint."""

    def __init__(self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        self._transport = transport
        self._request_id = 0

    async def list_tools(self) -> list[dict[str, Any]]:
        data = await self._jsonrpc("tools/list", {})
        return list((data.get("result") or {}).get("tools") or [])

    async def health_check(self) -> MCPToolResult:
        return await self.call_tool("health_check", {})

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> MCPToolResult:
        try:
            data = await self._jsonrpc("tools/call", {"name": name, "arguments": arguments or {}})
            if "error" in data and data["error"]:
                return MCPToolResult(tool=name, ok=False, data=data, error=str(data["error"]))
            normalized = self._normalize_tool_result((data.get("result") or {}))
            return MCPToolResult(tool=name, ok=True, data=normalized)
        except Exception as exc:  # noqa: BLE001 - preserve tool failure as data for the LLM.
            return MCPToolResult(tool=name, ok=False, data=None, error=str(exc))

    async def _jsonrpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": self.settings.mcp_protocol_version,
        }
        async with self._client() as client:
            response = await client.post("", headers=headers, json=payload)
            response.raise_for_status()
            return self._decode_response(response)

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.settings.mcp_server_url.rstrip("/"),
            timeout=httpx.Timeout(self.settings.mcp_timeout_seconds),
            verify=self.settings.mcp_verify_tls,
            transport=self._transport,
        )

    @staticmethod
    def _decode_response(response: httpx.Response) -> dict[str, Any]:
        text = response.text.strip()
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type or text.startswith("event:") or "\ndata:" in text:
            data_lines: list[str] = []
            for line in text.splitlines():
                if line.startswith("data:"):
                    data_lines.append(line.removeprefix("data:").strip())
            if not data_lines:
                raise MCPClientError("MCP SSE response contained no data lines.")
            text = data_lines[-1]
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise MCPClientError(f"MCP response was not valid JSON: {text[:300]}") from exc

    @staticmethod
    def _normalize_tool_result(result: dict[str, Any]) -> Any:
        """Normalize common MCP content formats into JSON-like Python data."""
        content = result.get("content")
        if isinstance(content, list):
            normalized_items: list[Any] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = str(item.get("text") or "")
                    normalized_items.append(_maybe_json(text))
                else:
                    normalized_items.append(item)
            if len(normalized_items) == 1:
                return normalized_items[0]
            return normalized_items
        if "structuredContent" in result:
            return result["structuredContent"]
        return result


def _maybe_json(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped[0] in "[{\"":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return text
    return text
