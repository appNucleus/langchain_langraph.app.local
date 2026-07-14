from __future__ import annotations

from app.schemas.execution import ExecutionBudget


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
