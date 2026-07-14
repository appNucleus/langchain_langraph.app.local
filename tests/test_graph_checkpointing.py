from __future__ import annotations

import pytest

from app.graph import ChatAgent
from app.schemas.chat import ChatRequest
from app.schemas.execution import ExecutionBudget
from app.settings import Settings


@pytest.mark.asyncio
async def test_echo_graph_preserves_execution_meter_and_completes() -> None:
    """The graph must keep request metering available without checkpointing locks."""

    settings = Settings(
        llm_backend="echo",
        mcp_enabled=False,
        state_backend="memory",
        checkpoint_backend="memory",
        artifact_backend="disabled",
    )
    agent = ChatAgent(settings)

    try:
        response = await agent.ainvoke(
            ChatRequest(
                message="hello",
                thread_id="execution-meter-regression-test",
            )
        )
    finally:
        await agent.aclose()

    assert response.backend == "echo"
    assert response.metadata["runtime_contract"] == "agent-graph-v1"
    assert "Message received" in response.response
    assert response.metadata["usage"]["model_calls"] == 0
    assert response.metadata["usage"]["tool_calls"] == 0
    assert response.metadata["usage"]["verifier_rounds"] == 1


def test_serialized_meter_snapshot_rehydrates_execution_budget() -> None:
    """Durable state stores data only; runtime services are reconstructed."""

    budget = ExecutionBudget(
        max_duration_seconds=60,
        max_model_calls=10,
        max_tool_calls=10,
        max_verifier_rounds=10,
    )
    budget.model_calls = 1
    budget.tool_calls = 2
    budget.verifier_rounds = 3

    serialized = budget.snapshot().model_dump(mode="json")
    restored = ExecutionBudget(
        max_duration_seconds=60,
        max_model_calls=10,
        max_tool_calls=10,
        max_verifier_rounds=10,
        state=serialized,
    )

    assert restored.model_calls == 1
    assert restored.tool_calls == 2
    assert restored.verifier_rounds == 3
    assert restored.started_at == pytest.approx(budget.started_at, abs=0.001)
    assert restored.elapsed_seconds >= budget.snapshot().active_execution_seconds
