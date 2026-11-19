"""Exact, additive token costing shared by the compressors and the renderer.

The allocator, compressors and renderer must agree on what an item costs, or
the overflow guarantee is only a hope. Every fragment is joined with
whitespace, and the counter splits on whitespace, so the cost of a rendered
section equals the sum of its fragments' costs. Message framing is charged at
a fixed :data:`MESSAGE_OVERHEAD` per message, which is an upper bound because
the renderer merges adjacent same-role messages.
"""
from __future__ import annotations

import json
from typing import Callable
from xml.sax.saxutils import quoteattr

from aikit import count_tokens

from .models import BlockKind, ContextItem

Counter = Callable[[str], int]

BASE_OVERHEAD = 3       # per request framing
MESSAGE_OVERHEAD = 4    # per message (role + delimiters)
SECTION_SEP = "\n\n"

_WRAPPERS: dict[BlockKind, tuple[str, str]] = {
    BlockKind.MEMORY: ("<memories>", "</memories>"),
    BlockKind.RETRIEVAL: ("<documents>", "</documents>"),
    BlockKind.TOOL_RESULT: ("<tool_results>", "</tool_results>"),
}

TRUNCATION_MARK = " … [truncated] … "


def tool_json(tool: dict) -> str:
    """Canonical serialisation used both for costing and for counting payloads."""
    return json.dumps(tool, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _attrs(item: ContextItem, key: str) -> str:
    parts = [f"{key}={quoteattr(item.id)}"]
    parts += [f"{k}={quoteattr(str(v))}" for k, v in sorted(item.metadata.items())]
    return " ".join(parts)


class Formatter:
    """Turns items into text fragments and prices them with ``counter``."""

    def __init__(self, counter: Counter = count_tokens, cache_size: int = 50_000):
        self.counter = counter
        self._cache: dict[str, int] = {}
        self._cache_size = cache_size

    def cost(self, text: str) -> int:
        """Memoised ``counter(text)`` (compressors re-price the same fragments often)."""
        c = self._cache.get(text)
        if c is None:
            c = self.counter(text)
            if len(self._cache) >= self._cache_size:
                self._cache.clear()
            self._cache[text] = c
        return c

    # ------------------------------------------------------------ fragments
    def fragment(self, kind: BlockKind, item: ContextItem) -> str:
        if kind == BlockKind.MEMORY:
            return f"- {item.text}"
        if kind == BlockKind.RETRIEVAL:
            return f"<document {_attrs(item, 'id')}>\n{item.text}\n</document>"
        if kind == BlockKind.TOOL_RESULT:
            return f"<tool_result {_attrs(item, 'name')}>\n{item.text}\n</tool_result>"
        if kind == BlockKind.TOOL_DEFS:
            return tool_json(item.tool or {})
        return item.text  # SYSTEM, HISTORY, QUERY

    def wrapper(self, kind: BlockKind) -> tuple[str, str] | None:
        return _WRAPPERS.get(kind)

    # ------------------------------------------------------------ costs
    def item_cost(self, kind: BlockKind, item: ContextItem) -> int:
        cost = self.cost(self.fragment(kind, item))
        if kind == BlockKind.HISTORY:
            cost += MESSAGE_OVERHEAD
        return cost

    def wrapper_cost(self, kind: BlockKind) -> int:
        w = self.wrapper(kind)
        return self.counter(w[0]) + self.counter(w[1]) if w else 0

    def block_cost(self, kind: BlockKind, items: list[ContextItem]) -> int:
        if not items:
            return 0
        return self.wrapper_cost(kind) + sum(self.item_cost(kind, it) for it in items)

    def text_overhead(self, kind: BlockKind, item: ContextItem) -> int:
        """Cost of the item with empty text (tags, attributes, framing)."""
        return self.item_cost(kind, item.model_copy(update={"text": ""}))

    def render_block(self, kind: BlockKind, items: list[ContextItem]) -> str:
        """Render a non-history, non-tool block as one whitespace-joined string."""
        if not items:
            return ""
        body = "\n".join(self.fragment(kind, it) for it in items)
        w = self.wrapper(kind)
        return f"{w[0]}\n{body}\n{w[1]}" if w else body


# ---------------------------------------------------------------- text fitting
def _boundaries(text: str) -> list[int]:
    """Candidate cut points: positions just after each whitespace-delimited word."""
    cuts, in_word = [], False
    for i, ch in enumerate(text):
        if ch.isspace():
            if in_word:
                cuts.append(i)
            in_word = False
        else:
            in_word = True
    if in_word:
        cuts.append(len(text))
    return cuts


def fit_prefix(text: str, budget: int, counter: Counter, suffix: str = " …") -> str:
    """Longest word-boundary prefix (+suffix) whose cost is <= budget.

    Works for any monotone counter (binary search over cut points).
    """
    if budget <= 0:
        return ""
    if counter(text) <= budget:
        return text
    cuts = _boundaries(text)
    lo, hi, best = 0, len(cuts) - 1, ""
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = text[: cuts[mid]].rstrip() + suffix
        if counter(cand) <= budget:
            best, lo = cand, mid + 1
        else:
            hi = mid - 1
    if not best and counter(suffix.strip()) <= budget:
        return ""
    return best


def fit_suffix(text: str, budget: int, counter: Counter) -> str:
    """Longest word-boundary suffix whose cost is <= budget."""
    if budget <= 0:
        return ""
    if counter(text) <= budget:
        return text
    starts = [0] + [i + 1 for i, ch in enumerate(text) if ch.isspace()]
    lo, hi, best = 0, len(starts) - 1, ""
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = text[starts[mid]:].lstrip()
        if counter(cand) <= budget:
            best, hi = cand, mid - 1
        else:
            lo = mid + 1
    return best


def fit_middle(text: str, budget: int, counter: Counter, head_share: float = 0.6) -> str:
    """Keep the head and tail of ``text`` around a truncation mark, within budget."""
    if counter(text) <= budget:
        return text
    room = budget - counter(TRUNCATION_MARK)
    if room <= 2:
        return fit_prefix(text, budget, counter)
    head = fit_prefix(text, int(room * head_share), counter, suffix="")
    tail = fit_suffix(text[len(head):], room - counter(head), counter)
    out = head + TRUNCATION_MARK + tail
    while counter(out) > budget and tail:  # defensive: non-additive custom counters
        tail = fit_suffix(tail, counter(tail) - 1, counter)
        out = head + TRUNCATION_MARK + tail
    return out if counter(out) <= budget else fit_prefix(text, budget, counter)


def fit_item(
    formatter: Formatter, kind: BlockKind, item: ContextItem, budget: int, mode: str = "end"
) -> ContextItem | None:
    """Shrink an item's text so its full cost fits ``budget``; None if impossible."""
    if formatter.item_cost(kind, item) <= budget:
        return item
    room = budget - formatter.text_overhead(kind, item)
    if room <= 0:
        return None
    fit = fit_middle if mode == "middle" else fit_prefix
    text = fit(item.text, room, formatter.counter)
    if not text.strip():
        return None
    new = item.model_copy(update={"text": text})
    return new if formatter.item_cost(kind, new) <= budget else None


def water_fill_items(budget: int, costs: dict[str, int]) -> dict[str, int]:
    """Max-min fair split of ``budget`` across items, never above each item's cost."""
    from .allocator import water_fill

    return water_fill(max(0, budget), costs, {k: 1.0 for k in costs})
