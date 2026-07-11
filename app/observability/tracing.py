from __future__ import annotations
from contextlib import contextmanager
from time import monotonic
from app.observability.metrics import metrics

@contextmanager
def span(name: str):
    started=monotonic()
    try:
        yield
        metrics.inc(f'{name}.ok')
    except Exception:
        metrics.inc(f'{name}.error')
        raise
    finally:
        metrics.observe(f'{name}.seconds', monotonic()-started)
