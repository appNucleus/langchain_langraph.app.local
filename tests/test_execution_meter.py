from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json

import pytest

from app.graphs.state import AgentGraphState
from app.orchestration.execution_meter import (
    BudgetExceeded,
    ExecutionBudget,
    ExecutionMeterState,
    model_operation_scope,
)


@pytest.mark.asyncio
async def test_primary_and_fallback_are_two_physical_attempts() -> None:
    budget = ExecutionBudget(60, 4, 4, 2)
    with model_operation_scope(budget):
        await budget.begin_model_attempt()
        await budget.finish_model_attempt(success=False)
        await budget.begin_model_attempt()
        await budget.finish_model_attempt(success=True)

    usage = budget.usage_metadata()
    assert usage["logical_model_operations"] == 1
    assert usage["physical_model_attempts"] == 2
    assert usage["fallback_attempts"] == 1
    assert usage["model_failures"] == 1
    assert usage["model_successes"] == 1


@pytest.mark.asyncio
async def test_failed_attempt_consumes_the_budget() -> None:
    budget = ExecutionBudget(60, 1, 1, 1)
    await budget.begin_model_attempt()
    await budget.finish_model_attempt(success=False)
    with pytest.raises(BudgetExceeded, match="model call budget"):
        await budget.begin_model_attempt()


@pytest.mark.asyncio
async def test_meter_state_preserves_absolute_deadline_and_elapsed_time() -> None:
    now = datetime.now(UTC)
    state = ExecutionMeterState(
        started_at=now - timedelta(seconds=8),
        deadline_at=now + timedelta(seconds=30),
        active_execution_seconds=3.5,
    )
    budget = ExecutionBudget(60, 4, 4, 2, state=state)
    snapshot = budget.snapshot()
    assert snapshot.deadline_at == state.deadline_at
    assert snapshot.active_execution_seconds >= 3.5
    assert snapshot.elapsed_wall_seconds >= 8


def test_checkpoint_state_contains_only_serialized_meter_snapshot() -> None:
    budget = ExecutionBudget(60, 4, 4, 2)
    snapshot = budget.snapshot().model_dump(mode="json")

    assert "execution_budget" not in AgentGraphState.__annotations__
    assert "execution_meter_state" in AgentGraphState.__annotations__
    assert json.loads(json.dumps(snapshot))["schema_version"] == 2
