from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedToolResult:
    ok: bool
    data: Any
    error: str | None = None
    raw: dict[str, Any] | None = None


def parse_tool_result(result: Any) -> ParsedToolResult:
    """Normalize an MCP tools/call result.

    ``structuredContent`` has precedence over textual content. ``isError=true``
    always produces ``ok=False`` even when useful structured error data exists.
    """

    if not isinstance(result, dict):
        return ParsedToolResult(ok=True, data=result, raw=None)

    is_error = result.get("isError") is True
    data = _normalize(result)
    error = _extract_error(data) if is_error else None
    return ParsedToolResult(ok=not is_error, data=data, error=error, raw=result)


def _normalize(result: dict[str, Any]) -> Any:
    if "structuredContent" in result:
        return result["structuredContent"]

    content = result.get("content")
    if isinstance(content, list):
        items: list[Any] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                items.append(_maybe_json(str(item.get("text") or "")))
            else:
                items.append(item)
        return items[0] if len(items) == 1 else items
    return result


def _maybe_json(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped[:1] in {"[", "{", '"'}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return text
    return text


def _extract_error(data: Any) -> str:
    if isinstance(data, dict):
        for key in ("error", "message", "detail"):
            value = data.get(key)
            if value:
                return str(value)
    if isinstance(data, list):
        return "; ".join(str(item) for item in data if item) or (
            "MCP tool returned isError=true."
        )
    return str(data) if data else "MCP tool returned isError=true."
