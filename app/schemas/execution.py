from __future__ import annotations
from pydantic import BaseModel

class ExecutionBudget(BaseModel):
    max_iterations: int = 4
    max_research_rounds: int = 2
    max_replans: int = 1

class ExecutionCounters(BaseModel):
    iterations: int = 0
    research_rounds: int = 0
    replans: int = 0
