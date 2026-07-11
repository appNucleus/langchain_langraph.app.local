from __future__ import annotations

from dataclasses import asdict, dataclass
from time import time
from typing import Any


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    event: str
    data: dict[str, Any]
    timestamp: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def event(name: str, **data: Any) -> dict[str, Any]:
    return ExecutionEvent(event=name, data=data, timestamp=time()).as_dict()
