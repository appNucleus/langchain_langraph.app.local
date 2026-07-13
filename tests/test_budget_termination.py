from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.graph import ChatAgent
from app.graphs.routes import after_budgeted_step, after_verification
from app.schemas.execution import BudgetExceeded, ExecutionBudget


def _agent() -> ChatAgent:
    agent = ChatAgent.__new__(ChatAgent)
    agent.settings = SimpleNamespace(
        llm_backend="ollama",
        model_general="qwen3.5:4b",
    )
    return agent


def _state(*, budget: ExecutionBudget) -> dict:
    return {
        "message": "diagnostic",
        "metadata": {"run_id": "budget-regression"},
        "plan": {
            "tasks": [
                {
                    "id": "t1",
                    "objective": "first task",
                    "completion_criteria": ["done"],
                },
                {
                    "id": "t2",
                    "objective": "second task",
                    "completion_criteria": ["done"],
                },
            ]
        },
        "task_index": 0,
        "task_results": [],
        "worker_result": {
            "answer": "A useful but not yet verified answer.",
            "claims": [],
            "confidence": 0.5,
        },
        "verification": {},
        "evidence": [],
        "iterations": 1,
        "research_rounds": 0,
        "replans": 0,
        "execution_budget": budget,
    }


def test_budget_rejects_model_calls_above_limit() -> None:
    budget = ExecutionBudget(60, 1, 2, 2)
    budget.model_calls = 2

    with pytest.raises(BudgetExceeded):
        budget.check()


def test_budget_uses_restart_safe_wall_clock() -> None:
    budget = ExecutionBudget(
        max_duration_seconds=60,
        max_model_calls=2,
        max_tool_calls=2,
        max_verifier_rounds=2,
    )

    assert budget.started_at > 1_000_000_000
    assert budget.elapsed_seconds >= 0
    budget.check()


@pytest.mark.asyncio
async def test_verifier_limit_routes_to_safe_termination_not_advance() -> None:
    agent = _agent()
    budget = ExecutionBudget(
        max_duration_seconds=60,
        max_model_calls=20,
        max_tool_calls=5,
        max_verifier_rounds=1,
    )
    budget.verifier_rounds = 1
    state = _state(budget=budget)

    updated = await agent._verify(state)

    assert updated["termination_reason"] == "maximum verifier rounds exceeded"
    assert updated["verification"]["verdict"] != "pass"
    assert updated["verification"]["task_complete"] is False
    assert after_verification(updated) == "terminate"

    terminated = await agent._terminate(updated)
    assert terminated["backend"] == "ollama"
    assert "not fully verified" in terminated["response"]
    assert "maximum verifier rounds exceeded" in terminated["response"]


@pytest.mark.asyncio
async def test_worker_precheck_returns_terminal_state_instead_of_raising() -> None:
    agent = _agent()
    budget = ExecutionBudget(
        max_duration_seconds=60,
        max_model_calls=20,
        max_tool_calls=5,
        max_verifier_rounds=1,
    )
    budget.verifier_rounds = 2
    state = _state(budget=budget)

    updated = await agent._worker(state)

    assert updated["termination_reason"] == "maximum verifier rounds exceeded"
    assert after_budgeted_step(updated) == "terminate"


@pytest.mark.asyncio
async def test_termination_preserves_completed_and_current_outputs() -> None:
    agent = _agent()
    budget = ExecutionBudget(60, 20, 5, 2)
    state = _state(budget=budget)
    state["termination_reason"] = "maximum verifier rounds exceeded"
    state["task_results"] = [
        {
            "worker_result": {"answer": "Verified result from task one."},
            "verification": {"verdict": "pass", "task_complete": True},
        }
    ]
    state["worker_result"] = {"answer": "Unverified result from task two."}

    terminated = await agent._terminate(state)

    assert "Verified result from task one." in terminated["response"]
    assert "Unverified result from task two." in terminated["response"]
    assert "Treat any in-progress output as unverified" in terminated["response"]
