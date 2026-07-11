"""Small, dependency-free in-memory metrics registry.

The registry is intentionally process-local for Phase 3. It is thread-safe,
bounded, and suitable for health diagnostics and lightweight operational
telemetry. A production metrics backend can replace it behind the same API.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from threading import Lock
from typing import Any


class MetricsRegistry:
    """Thread-safe counters and bounded timing samples."""

    def __init__(self, *, max_timing_samples: int = 1_000) -> None:
        if max_timing_samples < 1:
            raise ValueError("max_timing_samples must be at least 1")
        self._max_timing_samples = max_timing_samples
        self._lock = Lock()
        self._counters: Counter[str] = Counter()
        self._timings: defaultdict[str, list[float]] = defaultdict(list)

    def inc(self, name: str, amount: int = 1) -> None:
        if not name:
            raise ValueError("metric name must not be empty")
        with self._lock:
            self._counters[name] += amount

    def observe(self, name: str, value: float) -> None:
        if not name:
            raise ValueError("metric name must not be empty")
        with self._lock:
            values = self._timings[name]
            values.append(float(value))
            overflow = len(values) - self._max_timing_samples
            if overflow > 0:
                del values[:overflow]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            timings = {
                name: {
                    "count": len(values),
                    "avg": (sum(values) / len(values)) if values else 0.0,
                    "min": min(values) if values else 0.0,
                    "max": max(values) if values else 0.0,
                }
                for name, values in self._timings.items()
            }
            return {
                "counters": dict(self._counters),
                "timings": timings,
            }

    def reset(self) -> None:
        """Clear metrics. Intended for tests and controlled diagnostics."""
        with self._lock:
            self._counters.clear()
            self._timings.clear()


# Backward-compatible name used by the original Phase 3 package.
InMemoryMetrics = MetricsRegistry

# Shared process-local registry used by graph, tracing, and API endpoints.
metrics = MetricsRegistry()

__all__ = ["InMemoryMetrics", "MetricsRegistry", "metrics"]
