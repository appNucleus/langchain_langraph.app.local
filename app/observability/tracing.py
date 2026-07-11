from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import monotonic

from app.observability.metrics import metrics


@contextmanager
def span(name: str) -> Iterator[None]:
    started = monotonic()
    try:
        yield
    except Exception:
        metrics.inc(f"{name}.error")
        raise
    else:
        metrics.inc(f"{name}.ok")
    finally:
        metrics.observe(f"{name}.seconds", monotonic() - started)
