from __future__ import annotations
from pydantic import BaseModel, Field

class PlanTask(BaseModel):
    id: str
    objective: str
    required_evidence: list[str] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)

class ExecutionPlan(BaseModel):
    goal: str
    tasks: list[PlanTask] = Field(min_length=1)
