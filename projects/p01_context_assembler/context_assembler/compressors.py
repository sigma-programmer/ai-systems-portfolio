"""Per-strategy block compression.

Contract: ``compress(block, budget, ctx)`` returns items whose
``Formatter.block_cost`` is **<= budget** — every strategy re-verifies its
own output, so the assembler's guarantee does not depend on estimates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aikit import HashingEmbedder

from .formatting import Formatter, fit_item, water_fill_items
from .models import BlockKind, ContextBlock, ContextItem, Placement, Strategy
from .placement import order_items
from .summarize import Summarizer

SUMMARY_PREFIX = "[Summary of earlier conversation]"


@dataclass
class CompressionContext:
    """Shared, per-assembly state handed to every compressor."""

    formatter: Formatter
    embedder: HashingEmbedder
    summarizer: Summarizer
    query: str = ""
    _qvec: np.ndarray | None = None

    @property
    def query_vec(self) -> np.ndarray | None:
        if self._qvec is None and self.query:
            self._qvec = self.embedder.embed([self.query])[0]
        return self._qvec


@dataclass
class CompressionResult:
    """Items kept (possibly rewritten) plus a record of what happened."""

    items: list[ContextItem]
    dropped: list[str] = field(default_factory=list)
    compressed: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def lossy(self) -> bool:
        return bool(self.dropped or self.compressed)


def _ids(items: list[ContextItem]) -> list[str]:
    return [it.id for it in items]


# ------------------------------------------------------------------ helpers
def _cost(ctx: CompressionContext, block: ContextBlock, items: list[ContextItem]) -> int:
    return ctx.formatter.block_cost(block.kind, items)


def _item_budget(ctx: CompressionContext, block: ContextBlock, items: list[ContextItem], budget: int) -> int:
    """Budget left for one more item given the items already kept (wrapper included)."""
    base = _cost(ctx, block, items) if items else ctx.formatter.wrapper_cost(block.kind)
    return budget - base


def _finish(block: ContextBlock, original: list[ContextItem], kept: list[ContextItem], res: CompressionResult) -> CompressionResult:
    kept_ids = set(_ids(kept))
    res.items = kept
    res.dropped = [it.id for it in original if it.id not in kept_ids]
    return res


# ------------------------------------------------------------------ strategies
def compress_none(block: ContextBlock, budget: int, ctx: CompressionContext) -> CompressionResult:
    """Keep items in order, dropping trailing items only if forced."""
    kept: list[ContextItem] = []
    for it in block.items:
        if _cost(ctx, block, kept + [it]) <= budget:
            kept.append(it)
        else:
            break
    res = CompressionResult(items=[])
    if len(kept) < len(block.items):
        res.actions.append(f"dropped {len(block.items) - len(kept)} trailing item(s)")
    return _finish(block, block.items, kept, res)


def compress_drop(block: ContextBlock, budget: int, ctx: CompressionContext) -> CompressionResult:
    """All or nothing."""
    if _cost(ctx, block, block.items) <= budget:
        return CompressionResult(items=list(block.items))
    res = CompressionResult(items=[], actions=["block dropped (all-or-nothing)"])
    return _finish(block, block.items, [], res)


def compress_truncate(block: ContextBlock, budget: int, ctx: CompressionContext) -> CompressionResult:
    """Keep items in order; truncate the first one that does not fit and drop the rest."""
    kept: list[ContextItem] = []
    res = CompressionResult(items=[])
    for it in block.items:
        room = _item_budget(ctx, block, kept, budget)
        if ctx.formatter.item_cost(block.kind, it) <= room:
            kept.append(it)
            continue
        cut = fit_item(ctx.formatter, block.kind, it, room, mode="end")
        if cut is not None:
            kept.append(cut)
            res.compressed.append(it.id)
            res.actions.append(f"truncated {it.id!r} to {ctx.formatter.item_cost(block.kind, cut)} tokens")
        break
    return _finish(block, block.items, kept, res)


def compress_truncate_middle(block: ContextBlock, budget: int, ctx: CompressionContext) -> CompressionResult:
    """Give each item a fair share (water-filled by size) and keep its head and tail."""
    if _cost(ctx, block, block.items) <= budget:
        return CompressionResult(items=list(block.items))
    f = ctx.formatter
    res = CompressionResult(items=[])
    items = list(block.items)
    while items:
        room = budget - f.wrapper_cost(block.kind)
        shares = water_fill_items(room, {it.id: f.item_cost(block.kind, it) for it in items})
        kept: list[ContextItem] = []
        for it in items:
            cut = fit_item(f, block.kind, it, shares[it.id], mode="middle")
            if cut is None:
                break
            if cut.text != it.text:
                res.compressed.append(it.id)
            kept.append(cut)
        if len(kept) == len(items) and _cost(ctx, block, kept) <= budget:
            if res.compressed:
                res.actions.append(f"middle-truncated {len(res.compressed)} item(s)")
            return _finish(block, block.items, kept, res)
        # Shares too small for every item: drop the oldest (first) item and retry.
        res.compressed.clear()
        items = items[1:]
    res.actions.append("no item fits even when truncated")
    return _finish(block, block.items, [], res)


def _normalize_history(items: list[ContextItem], res: CompressionResult) -> list[ContextItem]:
    """The Messages API requires the first message to come from the user."""
    i = 0
    while i < len(items) and items[i].role != "user":
        i += 1
    if i:
        res.actions.append(f"dropped {i} leading assistant message(s) (user-first rule)")
    return items[i:]


def _recent_window(block: ContextBlock, items: list[ContextItem], budget: int, ctx: CompressionContext) -> list[ContextItem]:
    """Pinned messages plus the newest messages that fit, in chronological order.

    Costs are additive, so the window is grown with a running total (O(n)).
    """
    f = ctx.formatter
    pinned = [it for it in items if it.pinned]
    chosen = {it.id for it in pinned}
    used = _cost(ctx, block, pinned)
    if used > budget:
        chosen, used = set(), 0
    base = f.wrapper_cost(block.kind)
    for it in reversed(items):
        if it.id in chosen:
            continue
        add = f.item_cost(block.kind, it) + (base if not chosen else 0)
        if used + add <= budget:
            chosen.add(it.id)
            used += add
        else:
            break  # strict sliding window: never skip a newer message to fit an older one
    return [it for it in items if it.id in chosen]


def compress_drop_oldest(block: ContextBlock, budget: int, ctx: CompressionContext) -> CompressionResult:
    res = CompressionResult(items=[])
    items = _normalize_history(list(block.items), res)
    kept = _normalize_history(_recent_window(block, items, budget, ctx), res) if items else []
    if len(kept) < len(items):
        res.actions.append(f"dropped {len(items) - len(kept)} oldest message(s)")
    return _finish(block, block.items, kept, res)


def compress_summarize(block: ContextBlock, budget: int, ctx: CompressionContext) -> CompressionResult:
    """History: recent turns verbatim + a summary of the overflow. Other kinds: summarise all items."""
    f = ctx.formatter
    res = CompressionResult(items=[])
    if block.kind != BlockKind.HISTORY:
        if _cost(ctx, block, block.items) <= budget:
            return CompressionResult(items=list(block.items))
        stub = ContextItem(id=f"{block.name}:summary", text="", role=None, metadata={"summary": "true"})
        room = budget - f.wrapper_cost(block.kind) - f.text_overhead(block.kind, stub)
        text = ctx.summarizer.summarize([it.text for it in block.items], room, ctx.query) if room > 0 else ""
        kept = [stub.model_copy(update={"text": text})] if text else []
        if kept and _cost(ctx, block, kept) > budget:
            kept = []
        res.actions.append(f"summarized {len(block.items)} item(s) into {_cost(ctx, block, kept)} tokens")
        res.compressed = [it.id for it in block.items] if kept else []
        out = _finish(block, block.items, kept, res)
        out.dropped = [] if kept else out.dropped
        return out

    items = list(block.items)
    if _cost(ctx, block, items) <= budget:
        return _finish(block, block.items, _normalize_history(items, res), res)
    summary_budget = int(budget * block.summary_ratio)
    recent = _recent_window(block, items, budget - summary_budget, ctx)
    recent_ids = {it.id for it in recent}
    older = [it for it in items if it.id not in recent_ids]
    stub = ContextItem(id=f"{block.name}:summary", role="user", text=SUMMARY_PREFIX + "\n")
    room = budget - _cost(ctx, block, recent) - f.item_cost(block.kind, stub)
    kept = recent
    if older and room > 0:
        transcript = [f"{it.role.capitalize()}: {it.text}" for it in older]  # type: ignore[union-attr]
        text = ctx.summarizer.summarize(transcript, room, ctx.query)
        if text:
            summary = stub.model_copy(update={"text": f"{SUMMARY_PREFIX}\n{text}"})
            if _cost(ctx, block, [summary] + recent) <= budget:
                kept = [summary] + recent
                res.compressed = [it.id for it in older]
                res.actions.append(
                    f"summarized {len(older)} older message(s) into {f.item_cost(block.kind, summary)} tokens"
                )
    if kept is recent:  # no summary: the user-first rule applies to the window itself
        kept = _normalize_history(recent, res)
        if len(kept) < len(items):
            res.actions.append(f"dropped {len(items) - len(kept)} oldest message(s)")
    out = _finish(block, block.items, kept, res)
    out.dropped = [i for i in out.dropped if i not in set(out.compressed)]
    return out


def relevance_scores(block: ContextBlock, ctx: CompressionContext) -> dict[str, float]:
    """Blend of query similarity, upstream score, importance and recency, in [0, 1]-ish.

    weights: sim 0.55, upstream score 0.20 (max-normalised), importance 0.15,
    recency 0.10 (rank-normalised timestamps). Missing signals fall back to sim.
    """
    items = block.items
    if not items:
        return {}
    texts = [
        f"{it.tool.get('name', '')} {it.tool.get('description', '')}" if it.tool else it.text for it in items
    ]
    qv = ctx.query_vec
    sims = (ctx.embedder.embed(texts) @ qv).clip(min=0.0) if qv is not None else np.zeros(len(items))
    ups = [it.score for it in items if it.score is not None]
    max_up = max(ups) if ups and max(ups) > 0 else None
    stamps = sorted({it.timestamp for it in items if it.timestamp is not None})
    rank = {t: i / max(1, len(stamps) - 1) for i, t in enumerate(stamps)}
    out = {}
    for it, sim in zip(items, sims):
        up = (it.score / max_up) if (it.score is not None and max_up) else float(sim)
        rec = rank[it.timestamp] if it.timestamp is not None and len(stamps) > 1 else 0.5
        out[it.id] = round(0.55 * float(sim) + 0.20 * up + 0.15 * it.importance + 0.10 * rec, 6)
    return out


def compress_top_k(block: ContextBlock, budget: int, ctx: CompressionContext) -> CompressionResult:
    """Rerank by relevance (optionally MMR-diversified) and greedily pack the best items."""
    res = CompressionResult(items=[])
    scores = relevance_scores(block, ctx)
    res.scores = scores
    ranked = sorted(block.items, key=lambda it: (not it.pinned, -scores[it.id], it.id))
    if block.mmr_lambda is not None and len(ranked) > 1:
        ranked = _mmr(ranked, scores, block.mmr_lambda, ctx, picks=block.top_k)
    limit = block.top_k if block.top_k is not None else len(ranked)
    kept: list[ContextItem] = []
    for it in ranked:
        if len(kept) >= limit and not it.pinned:
            continue
        room = _item_budget(ctx, block, kept, budget)
        if ctx.formatter.item_cost(block.kind, it) <= room:
            kept.append(it)
        elif block.allow_partial and room >= block.min_partial_tokens:
            cut = fit_item(ctx.formatter, block.kind, it, room, mode="end")
            if cut is not None:
                kept.append(cut)
                res.compressed.append(it.id)
    if block.top_k is not None and len(block.items) > limit:
        res.actions.append(f"kept top-{limit} of {len(block.items)} by relevance")
    kept_ids = {it.id for it in kept}
    over = [it.id for it in block.items if it.id not in kept_ids]
    if over:
        res.actions.append(f"dropped {len(over)} lowest-ranked item(s)")
    if res.compressed:
        res.actions.append(f"truncated {len(res.compressed)} item(s) to fill remaining budget")
    kept = order_items(kept, scores, block.placement)
    return _finish(block, block.items, kept, res)


def _mmr(
    ranked: list[ContextItem], scores: dict[str, float], lam: float, ctx: CompressionContext, picks: int | None = None
) -> list[ContextItem]:
    """Maximal Marginal Relevance re-ordering (pinned items stay first).

    Vectorised: keeps a running max-similarity-to-selected array, so selecting
    ``picks`` items costs O(picks * n * d). Items beyond ``picks`` keep their
    relevance order (ranking strategies never look past ``top_k`` anyway).
    """
    pinned = [it for it in ranked if it.pinned]
    pool = [it for it in ranked if not it.pinned]
    if not pool:
        return pinned
    vecs = ctx.embedder.embed([it.text or it.id for it in pool])
    rel = np.array([scores[it.id] for it in pool])
    if pinned:
        pv = ctx.embedder.embed([it.text or it.id for it in pinned])
        max_sim = (vecs @ pv.T).max(axis=1)
    else:
        max_sim = np.zeros(len(pool))
    taken = np.zeros(len(pool), dtype=bool)
    order: list[int] = []
    for _ in range(min(len(pool), picks if picks is not None else len(pool))):
        mmr = lam * rel - (1 - lam) * max_sim
        mmr[taken] = -np.inf
        best = int(np.argmax(mmr))  # ties resolve to the earlier (higher-relevance) item
        order.append(best)
        taken[best] = True
        max_sim = np.maximum(max_sim, vecs @ vecs[best])
    order += [i for i in range(len(pool)) if not taken[i]]
    return pinned + [pool[i] for i in order]


COMPRESSORS = {
    Strategy.NONE: compress_none,
    Strategy.DROP: compress_drop,
    Strategy.TRUNCATE: compress_truncate,
    Strategy.TRUNCATE_MIDDLE: compress_truncate_middle,
    Strategy.DROP_OLDEST: compress_drop_oldest,
    Strategy.SUMMARIZE: compress_summarize,
    Strategy.TOP_K: compress_top_k,
}


def compress(block: ContextBlock, budget: int, ctx: CompressionContext) -> CompressionResult:
    """Dispatch to the block's strategy and enforce the cost contract."""
    if block.kind == BlockKind.HISTORY and block.strategy not in (Strategy.DROP_OLDEST, Strategy.SUMMARIZE, Strategy.DROP):
        fn = compress_drop_oldest  # other strategies would break message structure
    elif block.kind == BlockKind.TOOL_DEFS and block.strategy not in (Strategy.TOP_K, Strategy.DROP, Strategy.NONE):
        fn = compress_top_k  # schemas can be selected but never cut or summarised
    else:
        fn = COMPRESSORS[block.strategy]
    if block.kind == BlockKind.TOOL_DEFS:
        block = block.model_copy(update={"allow_partial": False})
    res = fn(block, max(0, budget), ctx)
    if block.kind == BlockKind.HISTORY and res.items and res.items[0].role != "user":
        res.items = _normalize_history(res.items, res)
        kept = {it.id for it in res.items}
        res.dropped = [it.id for it in block.items if it.id not in kept and it.id not in set(res.compressed)]
    if block.placement != Placement.ORIGINAL and block.strategy != Strategy.TOP_K and block.kind != BlockKind.HISTORY:
        res.items = order_items(res.items, relevance_scores(block, ctx), block.placement)
    if ctx.formatter.block_cost(block.kind, res.items) > budget:  # contract violated: fail safe
        res = CompressionResult(items=[], dropped=_ids(block.items), actions=["fail-safe: dropped block"])
    return res
