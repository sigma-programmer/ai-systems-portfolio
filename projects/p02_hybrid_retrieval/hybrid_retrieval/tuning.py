"""Reranker training and latency-vs-accuracy sweeps with a Pareto frontier."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .evaluation import evaluate
from .pipeline import HybridSearcher, SearchConfig
from .rerank import CrossEncoderReranker
from .types import Query


def build_training_set(
    searcher: HybridSearcher, queries: Sequence[Query], depth: int = 30, threshold: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Features/labels from first-stage hybrid candidates of labeled queries.

    Label 1 = grade >= threshold. Queries whose positives were not retrieved
    are still used (all-negative groups teach the model what *not* to promote).
    """
    cfg = SearchConfig(mode="hybrid", candidate_k=depth)
    Xs, ys = [], []
    for q in queries:
        qvec = searcher.embed_query(q.text)
        hits = searcher.candidates(q.text, cfg, q.filter, qvec)[:depth]
        if not hits:
            continue
        Xs.append(searcher.extractor.extract(q.text, qvec, [h.index for h in hits]))
        ys.append(np.array([1 if q.qrels.get(h.doc_id, 0) >= threshold else 0 for h in hits]))
    return np.vstack(Xs), np.concatenate(ys)


def train_reranker(
    searcher: HybridSearcher, train_queries: Sequence[Query], depth: int = 30, l2: float = 1.0
) -> CrossEncoderReranker:
    """Fit a CrossEncoderReranker on `train_queries` and install it on the searcher."""
    X, y = build_training_set(searcher, train_queries, depth)
    rr = CrossEncoderReranker(searcher.extractor).fit(X, y, l2=l2)
    searcher.set_reranker(rr)
    return rr


def default_sweep_configs() -> list[SearchConfig]:
    """The grid behind the Pareto plot: retriever type x index x pool size x rerank."""
    cfgs = [
        SearchConfig(mode="sparse", candidate_k=10),
        SearchConfig(mode="dense", candidate_k=10),
        SearchConfig(mode="dense", dense_index="ivf", nprobe=1, candidate_k=10),
        SearchConfig(mode="dense", dense_index="ivf", nprobe=4, candidate_k=10),
    ]
    for c in (10, 50):
        cfgs.append(SearchConfig(mode="hybrid", fusion="rrf", candidate_k=c))
        cfgs.append(SearchConfig(mode="hybrid", fusion="weighted", alpha=0.5, candidate_k=c))
    cfgs.append(SearchConfig(mode="hybrid", fusion="weighted", alpha=0.7, candidate_k=50))
    cfgs.append(SearchConfig(mode="hybrid", dense_index="ivf", nprobe=2, candidate_k=50))
    for depth in (10, 20, 40):
        cfgs.append(SearchConfig(mode="hybrid", candidate_k=50, rerank=True, rerank_depth=depth))
    for depth in (10, 20, 40):
        cfgs.append(
            SearchConfig(mode="hybrid", fusion="weighted", alpha=0.5, candidate_k=50, rerank=True, rerank_depth=depth)
        )
    cfgs.append(SearchConfig(mode="hybrid", candidate_k=10, rerank=True, rerank_depth=20))
    cfgs.append(SearchConfig(mode="hybrid", dense_index="ivf", nprobe=2, candidate_k=50, rerank=True, rerank_depth=40))
    return cfgs


def sweep(
    searcher: HybridSearcher, queries: Sequence[Query], configs: Sequence[SearchConfig] | None = None, k: int = 10
) -> list[dict]:
    return [evaluate(searcher, queries, c, k) for c in (configs or default_sweep_configs())]


def pareto_frontier(points: Sequence[dict], cost_key: str = "latency_ms_p50", gain_key: str = "ndcg@10") -> list[dict]:
    """Non-dominated points (lower cost, higher gain), sorted by cost.

    A point is dominated if another has cost <= and gain >= with one strict.
    """
    ordered = sorted(points, key=lambda p: (p[cost_key], -p[gain_key]))
    frontier: list[dict] = []
    best = -float("inf")
    for p in ordered:
        if p[gain_key] > best:
            frontier.append(p)
            best = p[gain_key]
    return frontier
