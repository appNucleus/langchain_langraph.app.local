from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.schemas.execution import ExecutionBudget
from app.tools.executor import ToolExecutionDenied, ToolExecutor


class FakeMcp:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        return {"ok": True}


def settings():
    return SimpleNamespace(mcp_max_concurrency=2, side_effect_policy_enabled=True)


@pytest.mark.asyncio
async def test_dynamic_read_only_tool_is_allowed() -> None:
    mcp = FakeMcp()
    executor = ToolExecutor(mcp, settings())
    budget = ExecutionBudget(30, 3, 3, 2)

    result = await executor.execute(
        "web_search_and_scrape",
        {"query": "test"},
        budget=budget,
        metadata={},
    )

    assert result == {"ok": True}
    assert mcp.calls == [("web_search_and_scrape", {"query": "test"})]
    assert budget.tool_calls == 1


@pytest.mark.asyncio
async def test_caller_metadata_cannot_approve_write_tool() -> None:
    executor = ToolExecutor(FakeMcp(), settings())
    budget = ExecutionBudget(30, 3, 3, 2)

    with pytest.raises(ToolExecutionDenied):
        await executor.execute(
            "mail_send_draft",
            {"draft_id": "1"},
            budget=budget,
            metadata={"approved_tools": ["mail_send_draft"]},
        )

    assert budget.tool_calls == 0


@pytest.mark.asyncio
async def test_ambiguous_unknown_tool_is_denied() -> None:
    executor = ToolExecutor(FakeMcp(), settings())
    budget = ExecutionBudget(30, 3, 3, 2)

    with pytest.raises(ToolExecutionDenied):
        await executor.execute("do_magic", {}, budget=budget, metadata={})
