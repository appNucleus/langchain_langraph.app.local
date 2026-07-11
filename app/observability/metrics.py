from __future__ import annotations
from collections import Counter, defaultdict
from threading import Lock
from typing import Any

class MetricsRegistry:
    def __init__(self) -> None:
        self._lock=Lock(); self._counters=Counter(); self._timings=defaultdict(list)
    def inc(self, name: str, amount: int = 1) -> None:
        with self._lock: self._counters[name]+=amount
    def observe(self, name: str, value: float) -> None:
        with self._lock:
            values=self._timings[name]; values.append(float(value))
            if len(values)>1000: del values[:-1000]
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            timings={k:{'count':len(v),'avg':sum(v)/len(v) if v else 0,'max':max(v) if v else 0} for k,v in self._timings.items()}
            return {'counters':dict(self._counters),'timings':timings}
metrics=MetricsRegistry()
