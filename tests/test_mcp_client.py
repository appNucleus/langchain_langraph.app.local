from __future__ import annotations

import json

import httpx
import pytest

from app.mcp.client import MCPClient
from app.mcp.errors import MCPResponseMismatchError
from app.settings import Settings


def _initialize_response(payload: dict, *, session_id: str = "session-123") -> httpx.Response:
    return httpx.Response(
        200,
        headers={"Mcp-Session-Id": session_id},
        json={
            "jsonrpc": "2.0",
            "id": payload["id"],
            "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test-mcp", "version": "1.0"},
            },
        },
    )


@pytest.mark.asyncio
async def test_mcp_initializes_once_and_reuses_session_for_tools() -> None:
    methods: list[str] = []
    session_headers: list[str | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        methods.append(payload["method"])
        session_headers.append(request.headers.get("Mcp-Session-Id"))
        if payload["method"] == "initialize":
            return _initialize_response(payload)
        if payload["method"] == "notifications/initialized":
            return httpx.Response(202)
        if payload["method"] == "tools/list":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": {"tools": [{"name": "health_check"}]}})
        raise AssertionError(payload)

    client = MCPClient(Settings(mcp_server_url="https://mcp.test/mcp"), transport=httpx.MockTransport(handler))
    assert await client.list_tools() == [{"name": "health_check"}]
    assert await client.list_tools() == [{"name": "health_check"}]
    assert methods == ["initialize", "notifications/initialized", "tools/list", "tools/list"]
    assert session_headers == [None, "session-123", "session-123", "session-123"]
    await client.aclose()


@pytest.mark.asyncio
async def test_mcp_call_tool_prefers_structured_content() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        if payload["method"] == "initialize":
            return _initialize_response(payload)
        if payload["method"] == "notifications/initialized":
            return httpx.Response(202)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {
                    "structuredContent": {"status": "canonical"},
                    "content": [{"type": "text", "text": json.dumps({"status": "fallback"})}],
                },
            },
        )

    client = MCPClient(Settings(mcp_server_url="https://mcp.test/mcp"), transport=httpx.MockTransport(handler))
    result = await client.call_tool("health_check", {})
    assert result.ok is True
    assert result.data == {"status": "canonical"}
    await client.aclose()


@pytest.mark.asyncio
async def test_mcp_is_error_becomes_failed_tool_result() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        if payload["method"] == "initialize":
            return _initialize_response(payload)
        if payload["method"] == "notifications/initialized":
            return httpx.Response(202)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {"isError": True, "content": [{"type": "text", "text": "provider unavailable"}]},
            },
        )

    client = MCPClient(Settings(mcp_server_url="https://mcp.test/mcp"), transport=httpx.MockTransport(handler))
    result = await client.call_tool("news_search", {"query": "test"})
    assert result.ok is False
    assert result.error == "provider unavailable"
    await client.aclose()


@pytest.mark.asyncio
async def test_mcp_multiline_sse_response_decode() -> None:
    settings = Settings(mcp_server_url="https://mcp.test/mcp", mcp_session_enabled=False)

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        encoded = json.dumps({"jsonrpc": "2.0", "id": payload["id"], "result": {"tools": []}})
        body = f": keepalive\nevent: message\ndata: {encoded}\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = MCPClient(settings, transport=httpx.MockTransport(handler))
    assert await client.list_tools() == []
    await client.aclose()


@pytest.mark.asyncio
async def test_mcp_response_id_mismatch_is_rejected() -> None:
    settings = Settings(mcp_server_url="https://mcp.test/mcp", mcp_session_enabled=False)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 999, "result": {"tools": []}})

    client = MCPClient(settings, transport=httpx.MockTransport(handler))
    with pytest.raises(MCPResponseMismatchError):
        await client.list_tools()
    await client.aclose()


@pytest.mark.asyncio
async def test_mcp_client_posts_exact_configured_url_without_trailing_slash() -> None:
    seen_urls: list[str] = []
    settings = Settings(mcp_server_url="https://mcp.test/mcp", mcp_session_enabled=False)

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        payload = json.loads(request.content.decode())
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": {"tools": []}})

    client = MCPClient(settings, transport=httpx.MockTransport(handler))
    assert await client.list_tools() == []
    assert seen_urls == ["https://mcp.test/mcp"]
    await client.aclose()


@pytest.mark.asyncio
async def test_mcp_client_follows_307_redirect_over_network_dns_endpoint() -> None:
    seen_urls: list[str] = []
    settings = Settings(mcp_server_url="https://mcp.test/mcp", mcp_session_enabled=False)

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        if str(request.url) == "https://mcp.test/mcp":
            return httpx.Response(307, headers={"location": "http://mcp.test/mcp"})
        payload = json.loads(request.content.decode())
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": {"tools": [{"name": "health_check"}]}})

    client = MCPClient(settings, transport=httpx.MockTransport(handler))
    assert await client.list_tools() == [{"name": "health_check"}]
    assert seen_urls == ["https://mcp.test/mcp", "http://mcp.test/mcp"]
    await client.aclose()


@pytest.mark.asyncio
async def test_mcp_client_can_disable_redirect_following() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(307, headers={"location": "http://mcp.test/mcp"})

    client = MCPClient(
        Settings(mcp_server_url="https://mcp.test/mcp", mcp_follow_redirects=False, mcp_session_enabled=False),
        transport=httpx.MockTransport(handler),
    )
    result = await client.call_tool("news_search", {"query": "test"})
    assert result.ok is False
    assert "307" in (result.error or "")
    await client.aclose()

@pytest.mark.asyncio
async def test_mcp_expired_session_reinitializes_once() -> None:
    initialize_count = 0
    tool_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal initialize_count, tool_count
        payload = json.loads(request.content.decode())
        method = payload["method"]
        if method == "initialize":
            initialize_count += 1
            return _initialize_response(payload, session_id=f"session-{initialize_count}")
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/list":
            tool_count += 1
            if tool_count == 1:
                return httpx.Response(404, text="expired")
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": {"tools": []}})
        raise AssertionError(payload)

    client = MCPClient(Settings(mcp_server_url="https://mcp.test/mcp"), transport=httpx.MockTransport(handler))
    assert await client.list_tools() == []
    assert initialize_count == 2
    assert tool_count == 2
    assert client.session_id == "session-2"
    await client.aclose()
