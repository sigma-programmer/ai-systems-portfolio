"""Cost controls: per-tenant daily caps, spend-rate circuit breaker, backend breakers.

`CostTracker.decide(tenant)` returns one of:

* ``"ok"``        - route normally
* ``"downgrade"`` - tenant spent >= `soft_fraction` of its daily cap, *or* the global
                    spend-rate breaker is open → force the cheapest tier
* ``"reject"``    - tenant at/over its hard daily cap (kill-switch)

`SpendRateBreaker` watches $ spent in a sliding window; if the rate exceeds
`max_usd_per_window` it opens for `cooldown` seconds (runaway-loop protection).
`CircuitBreaker` is the classic closed → open → half-open state machine used per backend.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .clock import Clock, SystemClock

DAY = 86_400.0


class SpendRateBreaker:
    def __init__(self, max_usd_per_window: float, window: float = 60.0, cooldown: float = 120.0,
                 clock: Clock | None = None):
        self.max_usd = max_usd_per_window
        self.window = window
        self.cooldown = cooldown
        self.clock = clock or SystemClock()
        self._events: deque[tuple[float, float]] = deque()
        self.open_until = float("-inf")
        self.trips = 0

    def record(self, usd: float) -> None:
        now = self.clock.now()
        self._events.append((now, usd))
        if not self.is_open() and self.rate() > self.max_usd:
            self.open_until = now + self.cooldown
            self.trips += 1

    def rate(self) -> float:
        now = self.clock.now()
        while self._events and now - self._events[0][0] > self.window:
            self._events.popleft()
        return sum(u for _, u in self._events)

    def is_open(self) -> bool:
        return self.clock.now() < self.open_until


@dataclass
class TenantBudget:
    daily_cap_usd: float
    soft_fraction: float = 0.8


@dataclass
class CostTracker:
    budgets: dict[str, TenantBudget] = field(default_factory=dict)
    default_budget: TenantBudget | None = None
    rate_breaker: SpendRateBreaker | None = None
    clock: Clock = field(default_factory=SystemClock)

    def __post_init__(self) -> None:
        self._spend: dict[tuple[str, int], float] = {}
        self.total_usd = 0.0
        self.decisions: dict[str, int] = {"ok": 0, "downgrade": 0, "reject": 0}

    def _day(self) -> int:
        return int(self.clock.now() // DAY)

    def spent_today(self, tenant: str) -> float:
        return self._spend.get((tenant, self._day()), 0.0)

    def headroom(self, tenant: str) -> float | None:
        """USD left under the tenant's hard cap today (None = unlimited)."""
        budget = self.budgets.get(tenant, self.default_budget)
        return None if budget is None else budget.daily_cap_usd - self.spent_today(tenant)

    def record(self, tenant: str, usd: float) -> None:
        key = (tenant, self._day())
        self._spend[key] = self._spend.get(key, 0.0) + usd
        self.total_usd += usd
        if self.rate_breaker:
            self.rate_breaker.record(usd)

    def decide(self, tenant: str) -> str:
        budget = self.budgets.get(tenant, self.default_budget)
        decision = "ok"
        if budget is not None:
            spent = self.spent_today(tenant)
            if spent >= budget.daily_cap_usd:
                decision = "reject"
            elif spent >= budget.soft_fraction * budget.daily_cap_usd:
                decision = "downgrade"
        if decision == "ok" and self.rate_breaker and self.rate_breaker.is_open():
            decision = "downgrade"
        self.decisions[decision] += 1
        return decision


class CircuitBreaker:
    """Opens after `failure_threshold` consecutive failures; half-opens after `cooldown`."""

    def __init__(self, failure_threshold: int = 3, cooldown: float = 30.0,
                 clock: Clock | None = None):
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self.clock = clock or SystemClock()
        self.failures = 0
        self.opened_at: float | None = None
        self.trips = 0

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if self.clock.now() - self.opened_at >= self.cooldown:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def failure(self) -> None:
        self.failures += 1
        if self.state == "half_open" or self.failures >= self.failure_threshold:
            if self.opened_at is None or self.state == "half_open":
                self.trips += 1
            self.opened_at = self.clock.now()
