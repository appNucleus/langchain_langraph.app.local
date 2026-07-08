from __future__ import annotations

import json

import httpx
import pytest

from app.mcp.client import MCPClient
from app.settings import Settings


@pytest.mark.asyncio
async def test_mcp_list_tools_with_mock_transport() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        assert payload["method"] == "tools/list"
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": payload["id"], "result": {"tools": [{"name": "health_check"}]}},
        )

    client = MCPClient(Settings(mcp_server_url="https://mcp.test/mcp"), transport=httpx.MockTransport(handler))
    tools = await client.list_tools()
    assert tools == [{"name": "health_check"}]


@pytest.mark.asyncio
async def test_mcp_call_tool_normalizes_text_json() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        assert payload["method"] == "tools/call"
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {
                    "content": [
                        {"type": "text", "text": json.dumps({"status": "ok", "url": "https://example.com"})}
                    ]
                },
            },
        )

    client = MCPClient(Settings(mcp_server_url="https://mcp.test/mcp"), transport=httpx.MockTransport(handler))
    result = await client.call_tool("health_check", {})
    assert result.ok is True
    assert result.data["status"] == "ok"
    assert result.data["url"] == "https://example.com"


@pytest.mark.asyncio
async def test_mcp_sse_response_decode() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        body = "event: message\n" + "data: " + json.dumps({"jsonrpc": "2.0", "id": payload["id"], "result": {"tools": []}}) + "\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = MCPClient(Settings(mcp_server_url="https://mcp.test/mcp"), transport=httpx.MockTransport(handler))
    assert await client.list_tools() == []


@pytest.mark.asyncio
async def test_mcp_client_posts_exact_configured_url_without_trailing_slash() -> None:
    seen_urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        payload = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": payload["id"], "result": {"tools": []}},
        )

    client = MCPClient(Settings(mcp_server_url="https://mcp.test/mcp"), transport=httpx.MockTransport(handler))

    assert await client.list_tools() == []
    assert seen_urls == ["https://mcp.test/mcp"]


@pytest.mark.asyncio
async def test_mcp_client_follows_307_redirect_over_network_dns_endpoint() -> None:
    seen_urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        if str(request.url) == "https://mcp.test/mcp":
            return httpx.Response(307, headers={"location": "http://mcp.test/mcp"})
        payload = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": payload["id"], "result": {"tools": [{"name": "health_check"}]}},
        )

    client = MCPClient(Settings(mcp_server_url="https://mcp.test/mcp"), transport=httpx.MockTransport(handler))

    assert await client.list_tools() == [{"name": "health_check"}]
    assert seen_urls == ["https://mcp.test/mcp", "http://mcp.test/mcp"]


@pytest.mark.asyncio
async def test_mcp_client_can_disable_redirect_following() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(307, headers={"location": "http://mcp.test/mcp"})

    client = MCPClient(
        Settings(mcp_server_url="https://mcp.test/mcp", mcp_follow_redirects=False),
        transport=httpx.MockTransport(handler),
    )

    result = await client.call_tool("news_search", {"query": "test"})
    assert result.ok is False
    assert "HTTPStatusError" in (result.error or "")
    assert "307" in (result.error or "")
