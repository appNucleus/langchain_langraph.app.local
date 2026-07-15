from __future__ import annotations

from typing import Any

import pytest

from app.schemas.execution import ExecutionBudget
from app.settings import Settings
from app.tools.executor import ToolExecutionDenied, ToolExecutor


class _ToolResult:
    ok = True


class _FakeMCP:
    physical_attempts_metered = False

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> _ToolResult:
        self.calls.append((name, arguments))
        return _ToolResult()


def _executor() -> tuple[ToolExecutor, _FakeMCP, ExecutionBudget]:
    settings = Settings(_env_file=None, mcp_max_concurrency=1)
    mcp = _FakeMCP()
    return ToolExecutor(mcp, settings), mcp, ExecutionBudget(60, 2, 2, 1)


@pytest.mark.asyncio
async def test_caller_metadata_cannot_authorize_write_tool() -> None:
    executor, mcp, budget = _executor()

    with pytest.raises(ToolExecutionDenied):
        await executor.execute(
            "mail_create_draft",
            {"body": "do not send"},
            budget=budget,
            metadata={"allow_write_tools": True, "approved": True},
        )

    assert mcp.calls == []
    assert budget.tool_calls == 0


@pytest.mark.asyncio
async def test_ambiguous_or_unknown_tool_remains_fail_closed() -> None:
    executor, mcp, budget = _executor()

    with pytest.raises(ToolExecutionDenied):
        await executor.execute(
            "process_payload",
            {},
            budget=budget,
            metadata={"read_only": True},
        )

    assert mcp.calls == []
    assert budget.tool_calls == 0


@pytest.mark.asyncio
async def test_explicit_read_only_tool_still_executes_and_is_metered() -> None:
    executor, mcp, budget = _executor()

    result = await executor.execute(
        "web_search",
        {"query": "LangGraph persistence"},
        budget=budget,
        metadata={"allow_write_tools": False},
    )

    assert result.ok is True
    assert mcp.calls == [("web_search", {"query": "LangGraph persistence"})]
    assert budget.tool_calls == 1
    assert budget.state.tool_successes == 1
