"""Item ordering inside a block, to counter the lost-in-the-middle effect.

Long-context models attend best to the beginning and end of their input. With
``Placement.EDGES`` the best item goes first, the second-best last, the third
second, and so on, so the weakest material lands in the middle.
"""
from __future__ import annotations

from typing import TypeVar

from .models import ContextItem, Placement

T = TypeVar("T")


def edge_order(ranked: list[T]) -> list[T]:
    """[r1, r2, r3, r4, r5] -> [r1, r3, r5, r4, r2]."""
    front: list[T] = []
    back: list[T] = []
    for i, x in enumerate(ranked):
        (front if i % 2 == 0 else back).append(x)
    return front + back[::-1]


def order_items(items: list[ContextItem], scores: dict[str, float], placement: Placement) -> list[ContextItem]:
    if placement == Placement.ORIGINAL or len(items) < 2:
        return list(items)
    ranked = sorted(items, key=lambda it: (not it.pinned, -scores.get(it.id, 0.0), it.id))
    return ranked if placement == Placement.RELEVANCE else edge_order(ranked)
