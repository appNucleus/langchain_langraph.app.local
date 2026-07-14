from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime, timedelta
from typing import Iterator

from pydantic import BaseModel, ConfigDict, Field


class BudgetExceeded(RuntimeError):
    """Raised before an operation that would exceed the request budget."""


class ExecutionMeterState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    schema_version: int = 2
    logical_model_operations: int = 0
    physical_model_attempts: int = 0
    model_successes: int = 0
    model_failures: int = 0
    fallback_attempts: int = 0
    tool_attempts: int = 0
    tool_successes: int = 0
    tool_failures: int = 0
    tool_timeouts: int = 0
    verifier_rounds: int = 0
    revision_rounds: int = 0
    final_revision_rounds: int = 0
    research_rounds: int = 0
    replans: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    context_utilization: float | None = Field(default=None, ge=0)
    queue_wait_seconds: float = 0.0
    model_load_seconds: float = 0.0
    time_to_first_token: float | None = None
    checkpoint_reads: int = 0
    checkpoint_writes: int = 0
    checkpoint_bytes: int = 0
    artifact_reads: int = 0
    artifact_writes: int = 0
    started_at: datetime
    deadline_at: datetime
    elapsed_wall_seconds: float = 0.0
    active_execution_seconds: float = 0.0
    cancellation_count: int = 0


_CURRENT_METER: ContextVar[ExecutionBudget | None] = ContextVar(
    "execution_meter", default=None
)
_MODEL_OPERATION_ATTEMPTS: ContextVar[int | None] = ContextVar(
    "model_operation_attempts", default=None
)


class ExecutionBudget:
    """Request-scoped execution meter with a serializable state snapshot."""

    def __init__(
        self,
        max_duration_seconds: float,
        max_model_calls: int,
        max_tool_calls: int,
        max_verifier_rounds: int,
        *,
        state: ExecutionMeterState | dict[str, object] | None = None,
    ) -> None:
        self.max_duration_seconds = float(max_duration_seconds)
        self.max_model_calls = int(max_model_calls)
        self.max_tool_calls = int(max_tool_calls)
        self.max_verifier_rounds = int(max_verifier_rounds)
        now = datetime.now(UTC)
        if state is None:
            self.state = ExecutionMeterState(
                started_at=now,
                deadline_at=now + timedelta(seconds=self.max_duration_seconds),
            )
        else:
            self.state = ExecutionMeterState.model_validate(state)
        self._elapsed_before_resume = max(0.0, self.state.active_execution_seconds)
        self._started_monotonic = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def started_at(self) -> float:
        """Restart-safe wall-clock start time retained for the legacy budget API."""

        return self.state.started_at.timestamp()

    @property
    def model_calls(self) -> int:
        return self.state.physical_model_attempts

    @model_calls.setter
    def model_calls(self, value: int) -> None:
        self.state.physical_model_attempts = int(value)

    @property
    def tool_calls(self) -> int:
        return self.state.tool_attempts

    @tool_calls.setter
    def tool_calls(self, value: int) -> None:
        self.state.tool_attempts = int(value)

    @property
    def verifier_rounds(self) -> int:
        return self.state.verifier_rounds

    @verifier_rounds.setter
    def verifier_rounds(self, value: int) -> None:
        self.state.verifier_rounds = int(value)

    @property
    def elapsed_seconds(self) -> float:
        return self._elapsed_before_resume + max(
            0.0, time.monotonic() - self._started_monotonic
        )

    def remaining_seconds(self) -> float:
        return max(0.0, (self.state.deadline_at - datetime.now(UTC)).total_seconds())

    def check(self) -> None:
        if datetime.now(UTC) >= self.state.deadline_at:
            raise BudgetExceeded("execution deadline exceeded")
        if self.state.physical_model_attempts > self.max_model_calls:
            raise BudgetExceeded("model call budget exceeded")
        if self.state.tool_attempts > self.max_tool_calls:
            raise BudgetExceeded("tool call budget exceeded")
        if self.state.verifier_rounds > self.max_verifier_rounds:
            raise BudgetExceeded("maximum verifier rounds exceeded")

    def record_logical_model_operation(self) -> None:
        self.check()
        self.state.logical_model_operations += 1

    async def begin_model_attempt(self, *, fallback: bool | None = None) -> None:
        async with self._lock:
            self.check()
            if self.state.physical_model_attempts >= self.max_model_calls:
                raise BudgetExceeded("model call budget exceeded")
            operation_attempts = _MODEL_OPERATION_ATTEMPTS.get()
            inferred_fallback = (
                operation_attempts is not None and operation_attempts > 0
            )
            self.state.physical_model_attempts += 1
            if fallback is True or (fallback is None and inferred_fallback):
                self.state.fallback_attempts += 1
            if operation_attempts is not None:
                _MODEL_OPERATION_ATTEMPTS.set(operation_attempts + 1)

    async def finish_model_attempt(
        self,
        *,
        success: bool,
        prompt_tokens: int = 0,
        generated_tokens: int = 0,
        model_load_seconds: float = 0.0,
        time_to_first_token: float | None = None,
    ) -> None:
        async with self._lock:
            if success:
                self.state.model_successes += 1
            else:
                self.state.model_failures += 1
            self.state.prompt_tokens += max(0, int(prompt_tokens))
            self.state.generated_tokens += max(0, int(generated_tokens))
            self.state.model_load_seconds += max(0.0, float(model_load_seconds))
            if (
                time_to_first_token is not None
                and self.state.time_to_first_token is None
            ):
                self.state.time_to_first_token = max(0.0, float(time_to_first_token))

    async def begin_tool_attempt(self) -> None:
        async with self._lock:
            self.check()
            if self.state.tool_attempts >= self.max_tool_calls:
                raise BudgetExceeded("tool call budget exceeded")
            self.state.tool_attempts += 1

    async def finish_tool_attempt(
        self, *, success: bool, timed_out: bool = False
    ) -> None:
        async with self._lock:
            if success:
                self.state.tool_successes += 1
            else:
                self.state.tool_failures += 1
            if timed_out:
                self.state.tool_timeouts += 1

    def add_queue_wait(self, seconds: float) -> None:
        self.state.queue_wait_seconds += max(0.0, float(seconds))

    def record_cancellation(self) -> None:
        self.state.cancellation_count += 1

    def snapshot(self) -> ExecutionMeterState:
        snapshot = self.state.model_copy(deep=True)
        now = datetime.now(UTC)
        snapshot.elapsed_wall_seconds = max(
            0.0, (now - snapshot.started_at).total_seconds()
        )
        snapshot.active_execution_seconds = self.elapsed_seconds
        return snapshot

    def usage_metadata(self) -> dict[str, object]:
        snapshot = self.snapshot()
        return {
            **snapshot.model_dump(mode="json"),
            "model_calls": snapshot.physical_model_attempts,
            "tool_calls": snapshot.tool_attempts,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def get_current_execution_meter() -> ExecutionBudget | None:
    return _CURRENT_METER.get()


@contextmanager
def execution_meter_scope(meter: ExecutionBudget) -> Iterator[ExecutionBudget]:
    token: Token[ExecutionBudget | None] = _CURRENT_METER.set(meter)
    try:
        yield meter
    finally:
        _CURRENT_METER.reset(token)


@contextmanager
def model_operation_scope(meter: ExecutionBudget) -> Iterator[ExecutionBudget]:
    """Mark one logical model operation and infer internal fallback attempts."""

    meter.record_logical_model_operation()
    token: Token[int | None] = _MODEL_OPERATION_ATTEMPTS.set(0)
    try:
        yield meter
    finally:
        _MODEL_OPERATION_ATTEMPTS.reset(token)
