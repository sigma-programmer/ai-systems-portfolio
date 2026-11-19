"""Fault injection for testing and benchmarking durability.

The engine calls :meth:`FaultInjector.hit` at well-defined *crash points*:

``before_handler``  the attempt is checkpointed as started, handler not yet run
``after_effect``    a side effect ran but its idempotency record is not stored
``after_handler``   the handler returned but the step is not checkpointed
``after_commit``    the step checkpoint is durable

A :class:`SimulatedCrash` derives from ``BaseException`` so it is *not* caught
by the engine's ``except Exception`` handling. It unwinds the engine without
any cleanup (no lease release, no status change), exactly like a process
dying at that instruction. The alternative :class:`ExitAt` really terminates
the process with ``os._exit``.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

CRASH_POINTS = ("before_handler", "after_effect", "after_handler", "after_commit")


class SimulatedCrash(BaseException):
    """Raised to emulate an abrupt process death inside the engine."""


class FaultInjector:
    """No-op base injector; subclasses decide when to fail."""

    def hit(self, point: str, run_id: str, step: str, attempt: int) -> None:  # noqa: D401
        return None


@dataclass
class CrashAt(FaultInjector):
    """Crash at ``point`` (optionally only for ``step``) the first ``times`` times it is reached."""

    point: str
    step: str | None = None
    times: int = 1
    fired: int = 0

    def __post_init__(self) -> None:
        if self.point not in CRASH_POINTS:
            raise ValueError(f"unknown crash point {self.point!r}")

    def _matches(self, point: str, step: str) -> bool:
        return point == self.point and (self.step is None or self.step == step) and self.fired < self.times

    def hit(self, point: str, run_id: str, step: str, attempt: int) -> None:
        if self._matches(point, step):
            self.fired += 1
            raise SimulatedCrash(f"simulated crash at {point} in {step}")


@dataclass
class ExitAt(CrashAt):
    """Really kill the current process (``os._exit``) at the crash point."""

    exit_code: int = 137

    def hit(self, point: str, run_id: str, step: str, attempt: int) -> None:
        if self._matches(point, step):
            os._exit(self.exit_code)


@dataclass
class RandomFaults(FaultInjector):
    """Seeded random crashes at any crash point with probability ``crash_rate``.

    Each ``(run, step, attempt, point)`` site crashes at most once, so a run
    always makes progress eventually (like real-world sporadic failures).
    """

    crash_rate: float = 0.05
    seed: int = 0
    crashes: int = 0
    last_crash: tuple[str, str, int, str] | None = None
    _seen: set[tuple[str, str, int, str]] = field(default_factory=set)

    def hit(self, point: str, run_id: str, step: str, attempt: int) -> None:
        site = (run_id, step, attempt, point)
        if site in self._seen:
            return
        self._seen.add(site)
        rng = random.Random(f"{self.seed}:{run_id}:{step}:{attempt}:{point}")
        if rng.random() < self.crash_rate:
            self.crashes += 1
            self.last_crash = site
            raise SimulatedCrash(f"random crash at {point} in {step}")
