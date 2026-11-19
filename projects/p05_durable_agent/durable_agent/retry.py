"""Retry policy: exponential backoff with full jitter and error classification."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from .errors import FatalError, RetryableError


@dataclass(frozen=True)
class RetryPolicy:
    """How a step is retried.

    Delay before attempt ``n+1`` (after ``n`` failures) uses *full jitter*::

        cap   = min(max_delay, base_delay * multiplier ** (n - 1))
        delay = uniform(0, cap)

    Full jitter spreads retries of many clients evenly over the window, which
    avoids synchronized retry storms (see the AWS architecture blog's
    "Exponential Backoff and Jitter"). The random draw is seeded from
    ``(seed, run_id, step, attempt)`` so a given run's schedule is reproducible
    across processes and crashes.

    Classification order: ``fatal_on`` / :class:`FatalError` → fatal;
    ``retry_on`` / :class:`RetryableError` (incl. timeouts) → retryable;
    anything else → ``retry_unknown``.
    """

    max_attempts: int = 3
    base_delay: float = 0.5
    multiplier: float = 2.0
    max_delay: float = 30.0
    jitter: bool = True
    retry_on: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError)
    fatal_on: tuple[type[BaseException], ...] = (ValueError, TypeError, KeyError, PermissionError)
    retry_unknown: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay < 0 or self.max_delay < 0 or self.multiplier < 1:
            raise ValueError("delays must be >= 0 and multiplier >= 1")

    def is_retryable(self, exc: BaseException) -> bool:
        if isinstance(exc, FatalError):
            return False
        if isinstance(exc, RetryableError):
            return True
        if isinstance(exc, self.fatal_on):
            return False
        if isinstance(exc, self.retry_on):
            return True
        return self.retry_unknown

    def cap(self, failures: int) -> float:
        """Upper bound of the backoff window after ``failures`` failures (>= 1)."""
        exp = self.base_delay * (self.multiplier ** max(failures - 1, 0))
        return min(self.max_delay, exp)

    def delay(self, failures: int, run_id: str = "", step: str = "") -> float:
        """Backoff delay after ``failures`` consecutive failures."""
        cap = self.cap(failures)
        if not self.jitter:
            return cap
        material = f"{self.seed}:{run_id}:{step}:{failures}".encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))
        return rng.uniform(0.0, cap)


NO_RETRY = RetryPolicy(max_attempts=1)
