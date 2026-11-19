"""Working memory: a token-budgeted scratchpad with FIFO+importance overflow.

When the budget is exceeded we look only at the oldest `fifo_window` fraction
of items (FIFO pressure: recent turns are never evicted first) and, among
those, evict the one with the *lowest importance*. A salient old turn
("remember my flight is on Friday") therefore survives while older small talk
goes. Evicted items are returned to the caller, which promotes them to
episodic memory — nothing is silently lost.
"""
from __future__ import annotations

import math

from aikit import count_tokens

from .clock import Clock
from .importance import ImportanceScorer, salience_of
from .models import WorkingItem, new_id


class WorkingMemory:
    def __init__(self, clock: Clock, token_budget: int = 200, fifo_window: float = 0.5,
                 half_life: float = 1800.0, scorer: ImportanceScorer | None = None):
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if not 0 < fifo_window <= 1:
            raise ValueError("fifo_window must be in (0, 1]")
        self.clock = clock
        self.token_budget = token_budget
        self.fifo_window = fifo_window
        self.half_life = half_life
        self.scorer = scorer or ImportanceScorer()
        self.items: list[WorkingItem] = []
        self.evictions = 0

    @property
    def tokens(self) -> int:
        return sum(i.tokens for i in self.items)

    def importance(self, item: WorkingItem) -> float:
        return self.scorer.score(item.salience, item.last_access, item.access_count,
                                 self.clock.now(), self.half_life)

    def add(self, text: str, session_id: str, role: str = "user",
            salience: float | None = None) -> tuple[WorkingItem, list[WorkingItem]]:
        """Append a turn; returns (new item, items evicted to make room)."""
        now = self.clock.now()
        item = WorkingItem(new_id("wm"), session_id, role, text, now, now, count_tokens(text),
                           salience_of(text) if salience is None else salience)
        self.items.append(item)
        return item, self._enforce_budget(protect=item)

    def _enforce_budget(self, protect: WorkingItem) -> list[WorkingItem]:
        evicted: list[WorkingItem] = []
        while self.tokens > self.token_budget and len(self.items) > 1:
            candidates = [i for i in self.items if i is not protect]
            window = candidates[: max(1, math.ceil(len(candidates) * self.fifo_window))]
            victim = min(window, key=self.importance)
            self.items.remove(victim)
            evicted.append(victim)
        self.evictions += len(evicted)
        return evicted

    def touch(self, item: WorkingItem) -> None:
        item.access_count += 1
        item.last_access = self.clock.now()

    def expire(self, ttl: float) -> list[WorkingItem]:
        """Remove items idle longer than ttl seconds (caller promotes them)."""
        now = self.clock.now()
        gone = [i for i in self.items if now - i.last_access > ttl]
        self.items = [i for i in self.items if i not in gone]
        return gone

    def drain(self) -> list[WorkingItem]:
        out, self.items = self.items, []
        return out

    def render(self) -> str:
        return "\n".join(f"{i.role}: {i.text}" for i in self.items)
