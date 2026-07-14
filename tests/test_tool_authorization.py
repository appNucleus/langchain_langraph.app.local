from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.orchestration.execution_meter import ExecutionBudget
from app.tools.executor import ToolApprovalRequired, ToolExecutor


class MCP:
    async def call_tool(self, name, arguments):
        return SimpleNamespace(ok=True, name=name, arguments=arguments)


@pytest.mark.asyncio
async def test_caller_metadata_cannot_approve_write_tool(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.tools.executor.policy_for",
        lambda _name: SimpleNamespace(confirmation_required=True),
    )
    executor = ToolExecutor(
        MCP(),
        SimpleNamespace(mcp_max_concurrency=1, side_effect_policy_enabled=True),
    )
    with pytest.raises(ToolApprovalRequired):
        await executor.execute(
            "write_tool",
            {"value": 1},
            budget=ExecutionBudget(60, 2, 2, 1),
            metadata={"approved_tools": ["write_tool"]},
        )
