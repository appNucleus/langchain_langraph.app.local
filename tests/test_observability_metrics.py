from __future__ import annotations

from app.observability import InMemoryMetrics, MetricsRegistry


def test_in_memory_metrics_preserve_interface_and_bound_timing_samples() -> None:
    registry = InMemoryMetrics(max_timing_samples=2)

    assert isinstance(registry, MetricsRegistry)

    registry.inc("requests")
    registry.observe("latency", 1.0)
    registry.observe("latency", 2.0)
    registry.observe("latency", 3.0)
    snapshot = registry.snapshot()

    assert snapshot["counters"]["requests"] == 1
    assert snapshot["timings"]["latency"] == {
        "count": 2,
        "avg": 2.5,
        "min": 2.0,
        "max": 3.0,
    }
