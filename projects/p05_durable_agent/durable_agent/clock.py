"""Injectable time sources so tests and benchmarks never really sleep."""
from __future__ import annotations

import threading
import time
from typing import Protocol


class Clock(Protocol):
    """Wall-clock seconds plus a sleeper."""

    def now(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    """Real time."""

    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class FakeClock:
    """Deterministic clock: ``sleep`` advances time instantly and is recorded.

    Thread-safe so that it can be shared by an engine and timed-out worker
    threads.
    """

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self._t = float(start)
        self._lock = threading.Lock()
        self.sleeps: list[float] = []

    def now(self) -> float:
        with self._lock:
            return self._t

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            if seconds > 0:
                self._t += seconds

    def advance(self, seconds: float) -> None:
        """Move time forward without recording a sleep."""
        with self._lock:
            self._t += seconds
