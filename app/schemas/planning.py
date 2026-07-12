from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


def _ensure_list(value: Any) -> Any:
    """Normalize common LLM singleton/null variants before strict validation."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, tuple):
        return list(value)
    return value


class PlanTask(BaseModel):
    id: str
    objective: str
    required_evidence: list[str] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)

    @field_validator("id", mode="before")
    @classmethod
    def normalize_task_id(cls, value: Any) -> Any:
        """Accept integer JSON task IDs while preserving a string state contract."""

        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return value

    @field_validator("required_evidence", "completion_criteria", mode="before")
    @classmethod
    def normalize_string_lists(cls, value: Any) -> Any:
        """Accept one string when a local model emits a singleton list as a scalar."""

        return _ensure_list(value)

    @field_validator("depends_on", mode="before")
    @classmethod
    def normalize_dependencies(cls, value: Any) -> Any:
        """Normalize dependency containers and integer task references."""

        normalized = _ensure_list(value)
        if not isinstance(normalized, list):
            return normalized
        return [
            str(item) if isinstance(item, int) and not isinstance(item, bool) else item
            for item in normalized
        ]


class ExecutionPlan(BaseModel):
    goal: str
    tasks: list[PlanTask] = Field(min_length=1)
