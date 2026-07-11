from __future__ import annotations
from dataclasses import dataclass
from time import monotonic

class BudgetExceeded(RuntimeError):
    pass

@dataclass
class ExecutionBudget:
    max_duration_seconds: float
    max_model_calls: int
    max_tool_calls: int
    max_verifier_rounds: int
    started_at: float = 0.0
    model_calls: int = 0
    tool_calls: int = 0
    verifier_rounds: int = 0

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = monotonic()

    @property
    def elapsed_seconds(self) -> float:
        return monotonic() - self.started_at

    def check(self) -> None:
        if self.elapsed_seconds > self.max_duration_seconds:
            raise BudgetExceeded('maximum execution duration exceeded')
        if self.model_calls > self.max_model_calls:
            raise BudgetExceeded('maximum model calls exceeded')
        if self.tool_calls > self.max_tool_calls:
            raise BudgetExceeded('maximum tool calls exceeded')
        if self.verifier_rounds > self.max_verifier_rounds:
            raise BudgetExceeded('maximum verifier rounds exceeded')
