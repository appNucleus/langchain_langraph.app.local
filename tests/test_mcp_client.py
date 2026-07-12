from __future__ import annotations

import asyncio
import json
from collections import Counter

import httpx
import pytest

from app.mcp.client import MCPClient
from app.mcp.errors import MCPResponseMismatchError, MCPSessionError
from app.settings import Settings


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "mcp_enabled": True,
        "mcp_server_url": "http://mcp.test/mcp",
        "mcp_initialize_on_startup": False,
        "inventory_cache_ttl_seconds": 60,
        "inventory_stale_if_error_seconds": 300,
    }
    values.update(overrides)
    return Settings(**values)


def _json_request(request: httpx.Request) -> dict[str, object]:
    return json.loads(request.content.decode("utf-8"))


def _delete_or_none(request: httpx.Request) -> httpx.Response | None:
    if request.method == "DELETE":
        return httpx.Response(204)
    return None


@pytest.mark.asyncio
async def test_initialize_is_single_flight_and_tool_inventory_is_cached() -> None:
    calls: Counter[str] = Counter()

    async def handler(request: httpx.Request) -> httpx.Response:
        if response := _delete_or_none(request):
            return response
        payload = _json_request(request)
        method = str(payload["method"])
        calls[method] += 1
        if method == "initialize":
            await asyncio.sleep(0.01)
            return httpx.Response(
                200,
                headers={"Mcp-Session-Id": "session-1"},
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {"tools": {"listChanged": True}},
                        "serverInfo": {"name": "test", "version": "1"},
                    },
                },
            )
        if method == "notifications/initialized":
            assert request.headers["Mcp-Session-Id"] == "session-1"
            assert request.headers["MCP-Protocol-Version"] == "2025-06-18"
            return httpx.Response(202)
        if method == "tools/list":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "tools": [
                            {
                                "name": "health_check",
                                "description": "Health",
                                "inputSchema": {"type": "object"},
                            }
                        ]
                    },
                },
            )
        raise AssertionError(method)

    client = MCPClient(_settings(), transport=httpx.MockTransport(handler))
    await asyncio.gather(*(client.initialize() for _ in range(8)))
    first, second = await client.list_tools(), await client.list_tools()

    assert first == second
    assert first[0]["name"] == "health_check"
    assert calls["initialize"] == 1
    assert calls["notifications/initialized"] == 1
    assert calls["tools/list"] == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_force_refresh_bypasses_tool_cache() -> None:
    list_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal list_calls
        if response := _delete_or_none(request):
            return response
        payload = _json_request(request)
        method = payload["method"]
        if method == "initialize":
            return httpx.Response(
                200,
                headers={"Mcp-Session-Id": "s"},
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                    },
                },
            )
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/list":
            list_calls += 1
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "tools": [
                            {
                                "name": f"tool-{list_calls}",
                                "inputSchema": {"type": "object"},
                            }
                        ]
                    },
                },
            )
        raise AssertionError(method)

    client = MCPClient(_settings(), transport=httpx.MockTransport(handler))
    assert (await client.list_tools())[0]["name"] == "tool-1"
    assert (await client.list_tools(force_refresh=True))[0]["name"] == "tool-2"
    assert list_calls == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_sse_notifications_do_not_hide_matching_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if response := _delete_or_none(request):
            return response
        payload = _json_request(request)
        method = payload["method"]
        if method == "initialize":
            return httpx.Response(
                200,
                headers={"Mcp-Session-Id": "sse-session"},
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                    },
                },
            )
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/list":
            response = {
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {
                    "tools": [
                        {"name": "x", "inputSchema": {"type": "object"}}
                    ]
                },
            }
            body = (
                'event: message\n'
                'data: {"jsonrpc":"2.0","method":"notifications/progress","params":{"progress":1}}\n\n'
                f"event: message\ndata: {json.dumps(response)}\n\n"
            )
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                text=body,
            )
        raise AssertionError(method)

    client = MCPClient(_settings(), transport=httpx.MockTransport(handler))
    assert (await client.list_tools())[0]["name"] == "x"
    await client.aclose()


@pytest.mark.asyncio
async def test_mismatched_response_id_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _json_request(request)
        if payload["method"] == "tools/list":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": 99999,
                    "result": {"tools": []},
                },
            )
        raise AssertionError(payload["method"])

    client = MCPClient(
        _settings(mcp_session_enabled=False),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(MCPResponseMismatchError):
        await client.list_tools()
    await client.aclose()


@pytest.mark.asyncio
async def test_structured_content_precedes_text_and_is_error_is_preserved() -> None:
    call_number = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_number
        if response := _delete_or_none(request):
            return response
        payload = _json_request(request)
        if payload["method"] == "tools/call":
            call_number += 1
            result = (
                {
                    "structuredContent": {"value": 42},
                    "content": [{"type": "text", "text": "ignored"}],
                    "isError": False,
                }
                if call_number == 1
                else {
                    "structuredContent": {"error": "bad input"},
                    "content": [{"type": "text", "text": "also ignored"}],
                    "isError": True,
                }
            )
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": payload["id"], "result": result},
            )
        raise AssertionError(payload["method"])

    client = MCPClient(
        _settings(mcp_session_enabled=False),
        transport=httpx.MockTransport(handler),
    )
    ok = await client.call_tool("example", {})
    failed = await client.call_tool("example", {})

    assert ok.ok is True and ok.data == {"value": 42}
    assert failed.ok is False
    assert failed.data == {"error": "bad input"}
    assert failed.error == "bad input"
    await client.aclose()


@pytest.mark.asyncio
async def test_read_only_call_reinitializes_and_retries_after_session_expiration() -> None:
    calls: Counter[str] = Counter()
    session_number = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal session_number
        if response := _delete_or_none(request):
            return response
        payload = _json_request(request)
        method = str(payload["method"])
        calls[method] += 1
        if method == "initialize":
            session_number += 1
            return httpx.Response(
                200,
                headers={"Mcp-Session-Id": f"session-{session_number}"},
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                    },
                },
            )
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/list" and calls[method] == 1:
            return httpx.Response(404, text="expired")
        if method == "tools/list":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {"tools": []},
                },
            )
        raise AssertionError(method)

    client = MCPClient(_settings(), transport=httpx.MockTransport(handler))
    assert await client.list_tools() == []
    assert calls["initialize"] == 2
    assert calls["tools/list"] == 2
    assert client.session_id == "session-2"
    await client.aclose()


@pytest.mark.asyncio
async def test_tool_call_is_not_replayed_after_session_expiration() -> None:
    calls: Counter[str] = Counter()
    session_number = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal session_number
        if response := _delete_or_none(request):
            return response
        payload = _json_request(request)
        method = str(payload["method"])
        calls[method] += 1
        if method == "initialize":
            session_number += 1
            return httpx.Response(
                200,
                headers={"Mcp-Session-Id": f"session-{session_number}"},
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                    },
                },
            )
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/call":
            return httpx.Response(410, text="expired")
        raise AssertionError(method)

    client = MCPClient(_settings(), transport=httpx.MockTransport(handler))
    result = await client.call_tool("side_effect", {"value": 1})

    assert result.ok is False
    assert "not automatically replayed" in (result.error or "")
    assert calls["tools/call"] == 1
    assert calls["initialize"] == 2  # recover for subsequent requests only
    await client.aclose()


@pytest.mark.asyncio
async def test_unsupported_negotiated_protocol_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _json_request(request)
        if payload["method"] == "initialize":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": "2099-01-01",
                        "capabilities": {},
                    },
                },
            )
        raise AssertionError(payload["method"])

    client = MCPClient(_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(MCPSessionError):
        await client.initialize()
    await client.aclose()


@pytest.mark.asyncio
async def test_server_ping_request_in_sse_receives_jsonrpc_response() -> None:
    ping_responses: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if response := _delete_or_none(request):
            return response
        payload = _json_request(request)
        # A response to a server-originated request has no method.
        if "method" not in payload:
            ping_responses.append(payload)
            return httpx.Response(202)
        method = payload["method"]
        if method == "initialize":
            return httpx.Response(
                200,
                headers={"Mcp-Session-Id": "ping-session"},
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                    },
                },
            )
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/list":
            result = {
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {"tools": []},
            }
            body = (
                'data: {"jsonrpc":"2.0","id":"server-ping-1","method":"ping","params":{}}\n\n'
                f"data: {json.dumps(result)}\n\n"
            )
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                text=body,
            )
        raise AssertionError(method)

    client = MCPClient(_settings(), transport=httpx.MockTransport(handler))
    assert await client.list_tools() == []
    assert ping_responses == [
        {"jsonrpc": "2.0", "id": "server-ping-1", "result": {}}
    ]
    await client.aclose()
