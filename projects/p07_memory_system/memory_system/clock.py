"""Injectable clocks so TTL, decay and recency are testable without sleeping."""
from __future__ import annotations

import time
from typing import Protocol

DAY = 86_400.0
HOUR = 3_600.0


class Clock(Protocol):
    def now(self) -> float: ...


class SystemClock:
    """Wall-clock seconds since the epoch."""

    def now(self) -> float:
        return time.time()


class ManualClock:
    """A clock that only moves when told to."""

    def __init__(self, start: float = 1_700_000_000.0):
        self.t = float(start)

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float = 0.0, *, hours: float = 0.0, days: float = 0.0) -> float:
        self.t += seconds + hours * HOUR + days * DAY
        return self.t
