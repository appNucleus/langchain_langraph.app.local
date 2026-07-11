from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from collections.abc import Iterator


@dataclass
class Timer:
    started: float
    elapsed_seconds: float = 0.0


@contextmanager
def measure() -> Iterator[Timer]:
    timer = Timer(started=perf_counter())
    try:
        yield timer
    finally:
        timer.elapsed_seconds = perf_counter() - timer.started
