from __future__ import annotations

import pytest

from app.graph import ChatAgent
from app.schemas.chat import ChatRequest
from app.schemas.execution import ExecutionBudget
from app.settings import Settings


@pytest.mark.asyncio
async def test_echo_graph_preserves_execution_budget_and_completes() -> None:
    """Regression test for the plan-node missing execution-budget failure."""

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
                thread_id="execution-budget-regression-test",
            )
        )
    finally:
        await agent.aclose()

    assert response.backend == "echo"
    # Compatibility-only metadata from the base ChatAgent remains stable.
    assert response.metadata["phase"] == "4"
    assert "Message received" in response.response
    assert response.metadata["usage"]["model_calls"] == 0
    assert response.metadata["usage"]["tool_calls"] == 0
    assert response.metadata["usage"]["verifier_rounds"] == 1


@pytest.mark.asyncio
async def test_checkpoint_round_trip_restores_execution_budget_type() -> None:
    """Ensure strict checkpoint serialization can restore ExecutionBudget."""

    from langgraph.graph import END, START, StateGraph

    from app.graphs.state import AgentGraphState
    from app.state.runtime import StateRuntime

    settings = Settings(
        llm_backend="echo",
        mcp_enabled=False,
        state_backend="memory",
        checkpoint_backend="memory",
        artifact_backend="disabled",
    )
    runtime = StateRuntime(settings)
    budget = ExecutionBudget(
        max_duration_seconds=60,
        max_model_calls=10,
        max_tool_calls=10,
        max_verifier_rounds=10,
    )
    history = [{"role": "user", "content": "prior message"}]

    async def inspect_state(state: AgentGraphState) -> AgentGraphState:
        restored_budget = state.get("execution_budget")
        assert isinstance(restored_budget, ExecutionBudget)
        assert state.get("history") == history
        restored_budget.model_calls += 1
        return {"response": "ok", "execution_budget": restored_budget}

    builder = StateGraph(AgentGraphState)
    builder.add_node("inspect", inspect_state)
    builder.add_edge(START, "inspect")
    builder.add_edge("inspect", END)
    graph = builder.compile(checkpointer=runtime.checkpointer)
    config = {"configurable": {"thread_id": "strict-budget-round-trip"}}

    result = await graph.ainvoke(
        {
            "message": "test",
            "history": history,
            "execution_budget": budget,
        },
        config=config,
    )
    snapshot = await graph.aget_state(config)

    assert isinstance(result["execution_budget"], ExecutionBudget)
    assert result["execution_budget"].model_calls == 1
    assert isinstance(snapshot.values["execution_budget"], ExecutionBudget)
    assert snapshot.values["execution_budget"].model_calls == 1
    assert snapshot.values["history"] == history
