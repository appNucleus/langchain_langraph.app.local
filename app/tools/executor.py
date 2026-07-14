from __future__ import annotations

import asyncio
import time
from typing import Any

from app.schemas.execution import ExecutionBudget
from app.tools.policies import policy_for


class ToolApprovalRequired(PermissionError):
    pass


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
        del metadata  # Caller metadata is context only, never tool authorization.
        policy = policy_for(name)
        if self.settings.side_effect_policy_enabled and policy.confirmation_required:
            raise ToolApprovalRequired(
                f"{name} is write-capable and server-issued approval is not implemented"
            )

        wait_started = time.monotonic()
        try:
            async with asyncio.timeout(max(0.001, budget.remaining_seconds())):
                async with self._semaphore:
                    budget.add_queue_wait(time.monotonic() - wait_started)
                    return await self.mcp.call_tool(name, arguments)
        except TimeoutError:
            raise
