from __future__ import annotations

import pytest

from app.graph import ChatAgent
from app.schemas.chat import ChatRequest
from app.settings import Settings


@pytest.mark.integration
@pytest.mark.asyncio
async def test_echo_graph_runs_plan_worker_verify_and_finalize() -> None:
    settings = Settings(
        llm_backend="echo",
        mcp_enabled=False,
        state_backend="memory",
        checkpoint_backend="memory",
        artifact_backend="disabled",
    )
    agent = ChatAgent(settings)
    await agent.start()
    try:
        response = await agent.ainvoke(
            ChatRequest(
                message="Provide a brief deterministic integration-test response.",
                thread_id="ci-integration-echo",
            )
        )
    finally:
        await agent.aclose()

    assert response.thread_id == "ci-integration-echo"
    assert response.response
    assert response.backend == "echo"
    assert response.metadata["termination_reason"] is None
    assert response.metadata["verification"][0]["verification"]["verdict"] == "pass"
