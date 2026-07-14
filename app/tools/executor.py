from __future__ import annotations

import asyncio
import time
from typing import Any

from app.schemas.execution import ExecutionBudget
from app.tools.policies import policy_for


class ToolExecutionDenied(PermissionError):
    """Raised when server policy does not classify a tool as safe to execute."""


class ToolApprovalRequired(ToolExecutionDenied):
    """Backward-compatible specialization for known write-capable tools."""


_READ_ONLY_MARKERS = {
    "calculate",
    "convert",
    "explain",
    "extract",
    "fetch",
    "get",
    "health",
    "list",
    "lookup",
    "news",
    "query",
    "quote",
    "read",
    "scrape",
    "search",
    "status",
    "time",
    "weather",
}
_WRITE_MARKERS = {
    "approve",
    "create",
    "delete",
    "draft",
    "execute",
    "modify",
    "patch",
    "post",
    "publish",
    "remove",
    "run",
    "send",
    "trigger",
    "update",
    "upload",
    "write",
}


def _tool_name_tokens(name: str) -> set[str]:
    return {part for part in name.lower().replace("-", "_").split("_") if part}


def _is_explicitly_safe_read(name: str, policy: Any) -> bool:
    if bool(getattr(policy, "confirmation_required", False)):
        return False
    tokens = _tool_name_tokens(name)
    if tokens & _WRITE_MARKERS:
        return False
    return bool(tokens & _READ_ONLY_MARKERS)


class ToolExecutor:
    def __init__(self, mcp: Any, settings: Any) -> None:
        self.mcp = mcp
        self.settings = settings
        self._semaphore = asyncio.Semaphore(settings.mcp_max_concurrency)

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        budget: ExecutionBudget,
        metadata: dict[str, Any],
    ) -> Any:
        del metadata  # Caller metadata is context only, never authorization.
        policy = policy_for(name)
        if bool(getattr(policy, "confirmation_required", False)):
            raise ToolApprovalRequired(
                f"{name} is write-capable and server-issued approval is not implemented"
            )
        if not _is_explicitly_safe_read(name, policy):
            raise ToolExecutionDenied(
                f"{name} is not classified as an approved read-only tool"
            )

        wait_started = time.monotonic()
        manual_metering = not bool(
            getattr(self.mcp, "physical_attempts_metered", False)
        )
        try:
            async with asyncio.timeout(max(0.001, budget.remaining_seconds())):
                async with self._semaphore:
                    budget.add_queue_wait(time.monotonic() - wait_started)
                    if manual_metering:
                        await budget.begin_tool_attempt()
                    try:
                        result = await self.mcp.call_tool(name, arguments)
                    except asyncio.CancelledError:
                        budget.record_cancellation()
                        if manual_metering:
                            await budget.finish_tool_attempt(success=False)
                        raise
                    except TimeoutError:
                        if manual_metering:
                            await budget.finish_tool_attempt(
                                success=False,
                                timed_out=True,
                            )
                        raise
                    except Exception:
                        if manual_metering:
                            await budget.finish_tool_attempt(success=False)
                        raise
                    if manual_metering:
                        await budget.finish_tool_attempt(
                            success=bool(getattr(result, "ok", True))
                        )
                    return result
        except TimeoutError:
            raise
