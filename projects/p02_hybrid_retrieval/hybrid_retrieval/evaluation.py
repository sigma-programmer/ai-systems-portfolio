"""IR metrics (recall@k, MRR, nDCG@k) and an evaluation harness.

Binary metrics (recall, MRR) count a doc as relevant when its grade is at
least `threshold` (default 2 = "answers the question"); nDCG uses the full
graded judgments with exponential gain 2^rel - 1.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Sequence

import numpy as np

from .pipeline import HybridSearcher, SearchConfig
from .types import Query


def recall_at_k(ranked: Sequence[str], qrels: dict[str, int], k: int, threshold: int = 2) -> float:
    relevant = {d for d, g in qrels.items() if g >= threshold}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], qrels: dict[str, int], threshold: int = 2, k: int | None = None) -> float:
    for rank, d in enumerate(ranked[:k] if k else ranked, start=1):
        if qrels.get(d, 0) >= threshold:
            return 1.0 / rank
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], qrels: dict[str, int], k: int) -> float:
    gains = [(2 ** qrels.get(d, 0)) - 1 for d in ranked[:k]]
    ideal = sorted(((2**g) - 1 for g in qrels.values() if g > 0), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def _percentile(xs: list[float], p: float) -> float:
    return float(np.percentile(xs, p)) if xs else 0.0


def evaluate(
    searcher: HybridSearcher,
    queries: Sequence[Query],
    config: SearchConfig,
    k: int = 10,
    warmup: bool = True,
) -> dict:
    """Run every query through `config`; return aggregate + per-family metrics and latency."""
    if warmup and queries:
        searcher.search(queries[0].text, k, config, queries[0].filter)
    per_type: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    latencies: list[float] = []
    rows: list[tuple[float, float, float, float]] = []
    for q in queries:
        t = time.perf_counter()
        resp = searcher.search(q.text, k, config, q.filter)
        latencies.append((time.perf_counter() - t) * 1000)
        ranked = resp.doc_ids
        row = (
            recall_at_k(ranked, q.qrels, 1),
            recall_at_k(ranked, q.qrels, k),
            reciprocal_rank(ranked, q.qrels, k=k),
            ndcg_at_k(ranked, q.qrels, k),
        )
        rows.append(row)
        per_type[q.qtype].append(row)

    def agg(rs: list[tuple[float, float, float, float]]) -> dict[str, float]:
        a = np.array(rs) if rs else np.zeros((1, 4))
        return {
            "recall@1": round(float(a[:, 0].mean()), 4),
            f"recall@{k}": round(float(a[:, 1].mean()), 4),
            "mrr": round(float(a[:, 2].mean()), 4),
            f"ndcg@{k}": round(float(a[:, 3].mean()), 4),
        }

    return {
        "config": config.label,
        "n_queries": len(queries),
        **agg(rows),
        "latency_ms_mean": round(float(np.mean(latencies)) if latencies else 0.0, 4),
        "latency_ms_p50": round(_percentile(latencies, 50), 4),
        "latency_ms_p95": round(_percentile(latencies, 95), 4),
        "by_type": {t: agg(rs) for t, rs in sorted(per_type.items())},
    }
