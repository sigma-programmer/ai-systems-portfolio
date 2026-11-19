"""Injectable clock (seconds)."""
from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...


class SystemClock:
    def now(self) -> float:
        return time.monotonic()


class ManualClock:
    def __init__(self, start: float = 0.0):
        self.t = float(start)

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> float:
        self.t += seconds
        return self.t
