from __future__ import annotations
from dataclasses import dataclass, asdict
from time import time
from typing import Any

@dataclass(frozen=True)
class ExecutionEvent:
    event: str
    data: dict[str, Any]
    timestamp: float = 0.0
    def as_dict(self) -> dict[str, Any]:
        value=asdict(self)
        value['timestamp']=self.timestamp or time()
        return value

def event(name: str, **data: Any) -> dict[str, Any]:
    return ExecutionEvent(name, data, time()).as_dict()
