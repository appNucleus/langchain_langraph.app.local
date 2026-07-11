from __future__ import annotations
import asyncio
from typing import Any
from app.schemas.execution import ExecutionBudget
from app.tools.policies import policy_for

class ToolApprovalRequired(PermissionError): pass

class ToolExecutor:
    def __init__(self, mcp, settings) -> None:
        self.mcp=mcp; self.settings=settings; self._semaphore=asyncio.Semaphore(settings.mcp_max_concurrency)
    async def execute(self, name: str, arguments: dict[str,Any], *, budget: ExecutionBudget, metadata: dict[str,Any]) -> Any:
        policy=policy_for(name)
        if self.settings.side_effect_policy_enabled and policy.confirmation_required:
            approved=set(metadata.get('approved_tools') or [])
            if name not in approved: raise ToolApprovalRequired(f'{name} requires explicit approval')
        budget.tool_calls+=1; budget.check()
        async with self._semaphore:
            return await self.mcp.call_tool(name, arguments)
