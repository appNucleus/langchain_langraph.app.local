from __future__ import annotations

import asyncio
from typing import Any

from app.logging_config import log_kv
from app.schemas.execution import ExecutionBudget
from app.tools.policies import policy_for

import logging

logger = logging.getLogger(__name__)


class ToolApprovalRequired(PermissionError):
    """Backward-compatible error retained for older callers."""


class ToolExecutionDenied(PermissionError):
    """Raised when server policy denies a write-capable or ambiguous tool."""


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
        policy = policy_for(name)
        if not policy.read_only:
            # Client-controlled metadata such as approved_tools is intentionally
            # ignored. Write operations stay disabled until server-issued,
            # argument-bound, replay-safe approval exists.
            log_kv(
                logger,
                logging.WARNING,
                "tool_execution_denied",
                tool=name,
                side_effect_level=policy.level.value,
                known_policy=policy.known,
                argument_keys=",".join(sorted(arguments)),
            )
            raise ToolExecutionDenied(
                f"Tool {name!r} is not permitted by the read-only server policy"
            )

        budget.tool_calls += 1
        budget.check()
        async with self._semaphore:
            return await self.mcp.call_tool(name, arguments)
