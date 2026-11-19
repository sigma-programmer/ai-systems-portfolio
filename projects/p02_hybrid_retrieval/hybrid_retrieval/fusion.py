"""Rank fusion: Reciprocal Rank Fusion and weighted normalized-score fusion.

Inputs are named ranked lists of (doc_index, score). Outputs are fused
(doc_index, fused_score, per-source contribution) triples, sorted.
"""
from __future__ import annotations

from typing import Literal, Mapping, Sequence

Ranked = Sequence[tuple[int, float]]
Fused = list[tuple[int, float, dict[str, float]]]


def _sort(acc: dict[int, float], parts: dict[int, dict[str, float]]) -> Fused:
    return [(i, s, parts[i]) for i, s in sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))]


def reciprocal_rank_fusion(
    lists: Mapping[str, Ranked], k: int = 60, weights: Mapping[str, float] | None = None
) -> Fused:
    """RRF (Cormack et al. 2009): score(d) = sum_s w_s / (k + rank_s(d)), ranks 1-based.

    Score-scale free, so it needs no calibration between BM25 and cosine.
    """
    acc: dict[int, float] = {}
    parts: dict[int, dict[str, float]] = {}
    for name, ranked in lists.items():
        w = 1.0 if weights is None else weights.get(name, 1.0)
        for rank, (idx, _score) in enumerate(ranked, start=1):
            contrib = w / (k + rank)
            acc[idx] = acc.get(idx, 0.0) + contrib
            parts.setdefault(idx, {})[name] = contrib
    return _sort(acc, parts)


def normalize_scores(ranked: Ranked, method: Literal["minmax", "max", "zscore"] = "minmax") -> dict[int, float]:
    """Map raw scores of one list onto a comparable scale."""
    if not ranked:
        return {}
    scores = [s for _, s in ranked]
    if method == "max":
        top = max(scores)
        return {i: (s / top if top > 0 else 0.0) for i, s in ranked}
    if method == "zscore":
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        sd = var**0.5 or 1.0
        return {i: (s - mean) / sd for i, s in ranked}
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return {i: 1.0 for i, _ in ranked}
    return {i: (s - lo) / (hi - lo) for i, s in ranked}


def weighted_score_fusion(
    lists: Mapping[str, Ranked],
    weights: Mapping[str, float],
    method: Literal["minmax", "max", "zscore"] = "minmax",
) -> Fused:
    """Convex combination of per-list normalized scores; absent docs contribute 0."""
    acc: dict[int, float] = {}
    parts: dict[int, dict[str, float]] = {}
    for name, ranked in lists.items():
        w = weights.get(name, 0.0)
        for idx, ns in normalize_scores(ranked, method).items():
            acc[idx] = acc.get(idx, 0.0) + w * ns
            parts.setdefault(idx, {})[name] = w * ns
    for idx in acc:  # make sure every doc reports every source
        for name in lists:
            parts[idx].setdefault(name, 0.0)
    return _sort(acc, parts)
