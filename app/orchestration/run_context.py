from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class RunIdentity:
    conversation_id: str
    run_id: str
    execution_thread_id: str

    @classmethod
    def resolve(
        cls,
        *,
        conversation_id: str | None = None,
        run_id: str | None = None,
        legacy_thread_id: str | None = None,
    ) -> "RunIdentity":
        resolved_conversation_id = conversation_id or legacy_thread_id or str(uuid4())
        resolved_run_id = run_id or str(uuid4())
        return cls(
            conversation_id=resolved_conversation_id,
            run_id=resolved_run_id,
            execution_thread_id=f"{resolved_conversation_id}:{resolved_run_id}",
        )
