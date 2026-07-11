from __future__ import annotations
from app.schemas.execution import BudgetExceeded, ExecutionBudget

def termination_reason(budget: ExecutionBudget) -> str | None:
    try: budget.check()
    except BudgetExceeded as exc: return str(exc)
    return None
