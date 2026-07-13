from __future__ import annotations

from typing import Any

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
        """Return a PostgreSQL-safe root-graph configuration.

        LangGraph reserves ``checkpoint_ns`` for nested graph namespaces and
        normalizes a non-empty namespace back to the root namespace for a
        top-level invocation. The unique execution thread already isolates one
        run from every other run, so the root namespace must remain empty.

        ``run_id`` is intentionally stored as JSON-safe string metadata rather
        than as a top-level UUID object. PostgreSQL checkpointers persist
        checkpoint metadata as JSONB, and keeping every value primitive avoids
        serialization failures after a model node completes.
        """

        return {
            "configurable": {
                "thread_id": self.execution_thread_id,
            },
            "metadata": {
                "conversation_id": self.conversation_id,
                "run_id": self.run_id,
                "execution_thread_id": self.execution_thread_id,
                "checkpoint_namespace": self.checkpoint_namespace,
                "state_schema_version": self.state_schema_version,
            },
        }
