"""Token budget allocation: required floors -> optional floors -> weighted surplus.

The allocator is a pure function over :class:`BlockDemand` records, so it can
be tested exhaustively and reused for the post-compression redistribution
pass. All arithmetic is integral; the allocations always sum to at most the
available budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BlockDemand:
    """What one block wants from the budget.

    Attributes:
        name: block name.
        priority: higher is granted first when budget is scarce.
        required: hard-required blocks get their floor before any optional block.
        floor: the smallest useful allocation (0 = may be squeezed to nothing).
        cap: the most this block can use (its demand, clipped by ``max_tokens``).
        weight: share of surplus in proportional allocation.
    """

    name: str
    priority: int
    required: bool
    floor: int
    cap: int
    weight: float = 1.0


@dataclass
class Allocation:
    """Result of an allocation round."""

    budget: int
    tokens: dict[str, int]
    dropped: list[str] = field(default_factory=list)   # optional blocks whose floor did not fit
    warnings: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.tokens.values())


def _order(d: BlockDemand) -> tuple:
    return (-d.priority, d.name)


def proportional_split(amount: int, weights: dict[str, float]) -> dict[str, int]:
    """Split an integer amount by weight with largest-remainder rounding."""
    out = {k: 0 for k in weights}
    total_w = sum(w for w in weights.values() if w > 0)
    if amount <= 0 or not weights:
        return out
    if total_w <= 0:
        weights = {k: 1.0 for k in weights}
        total_w = float(len(weights))
    raw = {k: amount * max(w, 0.0) / total_w for k, w in weights.items()}
    for k, v in raw.items():
        out[k] = int(v)
    rest = amount - sum(out.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - int(raw[k])), -weights[k], k))[:rest]:
        out[k] += 1
    return out


def water_fill(amount: int, caps: dict[str, int], weights: dict[str, float]) -> dict[str, int]:
    """Distribute ``amount`` by weight without exceeding per-key caps.

    Keys that saturate release their unused share to the remaining keys, round
    after round, until the amount is spent or every key is saturated.
    """
    given = {k: 0 for k in caps}
    remaining = amount
    active = {k for k, c in caps.items() if c > 0}
    while remaining > 0 and active:
        share = proportional_split(remaining, {k: weights.get(k, 1.0) for k in active})
        progress = 0
        for k in sorted(active):
            take = min(share[k], caps[k] - given[k])
            given[k] += take
            progress += take
        remaining -= progress
        active = {k for k in active if given[k] < caps[k]}
        if progress == 0:  # all positive shares were 0 (zero weights): hand out one token at a time
            for k in sorted(active, key=lambda k: (-weights.get(k, 1.0), k)):
                if remaining == 0:
                    break
                given[k] += 1
                remaining -= 1
            active = {k for k in active if given[k] < caps[k]}
    return given


class BudgetAllocator:
    """Three-phase allocator.

    1. **Required floors.** Granted in priority order. If they do not fit, the
       budget is split across required blocks proportionally to their floors
       (an *emergency*: required blocks will be truncated) and a warning is raised.
    2. **Optional floors.** Granted in priority order while they fit; a block
       whose floor does not fit is dropped entirely (it never gets surplus).
    3. **Surplus.** Water-filled by weight across all surviving blocks, up to
       each block's cap, with redistribution of whatever saturated blocks leave.
    """

    def allocate(self, demands: list[BlockDemand], budget: int) -> Allocation:
        names = [d.name for d in demands]
        if len(set(names)) != len(names):
            raise ValueError("block names must be unique")
        budget = max(0, budget)
        tokens = {d.name: 0 for d in demands}
        alloc = Allocation(budget=budget, tokens=tokens)
        remaining = budget

        required = sorted((d for d in demands if d.required), key=_order)
        need = sum(min(d.floor, d.cap) for d in required)
        if need > remaining:
            split = water_fill(remaining, {d.name: min(d.floor, d.cap) for d in required},
                               {d.name: float(max(1, min(d.floor, d.cap))) for d in required})
            tokens.update(split)
            remaining -= sum(split.values())
            alloc.warnings.append(
                f"required blocks need {need} tokens but only {budget} are available; "
                "required content will be truncated"
            )
        else:
            for d in required:
                tokens[d.name] = min(d.floor, d.cap)
            remaining -= need

        eligible = [d for d in required]
        for d in sorted((d for d in demands if not d.required), key=_order):
            floor = min(d.floor, d.cap)
            if floor <= remaining:
                tokens[d.name] = floor
                remaining -= floor
                eligible.append(d)
            else:
                alloc.dropped.append(d.name)

        caps = {d.name: d.cap - tokens[d.name] for d in eligible}
        extra = water_fill(remaining, caps, {d.name: d.weight for d in eligible})
        for k, v in extra.items():
            tokens[k] += v
        assert alloc.total <= budget, "allocator invariant violated"
        return alloc
