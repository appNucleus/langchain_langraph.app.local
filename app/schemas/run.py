from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RunIdentity(BaseModel):
    """Normalized server-authoritative identity for one graph execution."""

    model_config = ConfigDict(frozen=True)

    conversation_id: str = Field(min_length=1, max_length=200)
    run_id: str = Field(min_length=36, max_length=36)
    execution_thread_id: str = Field(min_length=1, max_length=237)
    checkpoint_namespace: str = Field(min_length=1, max_length=100)
    state_schema_version: int = Field(ge=1)
    request_hash: str = Field(min_length=64, max_length=64)
    resume_requested: bool = False
    resumed: bool = False
    client_supplied_run_id: bool = False

    def langgraph_config(self) -> dict[str, Any]:
        return {
            "configurable": {
                "thread_id": self.execution_thread_id,
                "checkpoint_ns": self.checkpoint_namespace,
            },
            "run_id": UUID(self.run_id),
        }
