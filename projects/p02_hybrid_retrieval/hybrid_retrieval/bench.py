"""Benchmark: accuracy/latency sweep, Pareto frontier, IVF scaling, filtering.

Accuracy numbers are fully deterministic (seeded corpus, deterministic
indexes and training); latencies are wall-clock and vary by machine.
Run: ``.venv/bin/python -m hybrid_retrieval.bench`` from the project dir,
or import and call ``benchmark()``.
"""
from __future__ import annotations

import json
import time

import numpy as np

from aikit import MockLLM

from .corpus import load_benchmark, split_queries
from .dense import FlatIndex, IVFIndex
from .evaluation import evaluate
from .pipeline import HybridSearcher, SearchConfig
from .rerank import LLMReranker
from .tuning import default_sweep_configs, pareto_frontier, sweep, train_reranker
from .types import Query


def _ivf_scaling(n: int = 50_000, dim: int = 128, n_lists: int = 128, n_queries: int = 200, seed: int = 0) -> dict:
    """Recall of IVF vs. exact search on clustered synthetic vectors, per nprobe."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(200, dim))
    x = centers[rng.integers(0, 200, n)] + 1.6 * rng.normal(size=(n, dim))
    x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    q = centers[rng.integers(0, 200, n_queries)] + 1.6 * rng.normal(size=(n_queries, dim))  # fresh draws
    q = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)
    flat = FlatIndex(x)
    t = time.perf_counter()
    ivf = IVFIndex(x, n_lists=n_lists, seed=seed)
    build_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    truth = [{i for i, _ in flat.search(v, 10)} for v in q]
    flat_ms = (time.perf_counter() - t) * 1000 / n_queries
    rows = [{"nprobe": "exact", "recall@10": 1.0, "latency_ms": round(flat_ms, 4), "scanned_frac": 1.0}]
    for nprobe in (1, 2, 4, 8, 16, 32):
        scanned = 0
        res = []
        t = time.perf_counter()
        for v in q:
            res.append({i for i, _ in ivf.search(v, 10, nprobe=nprobe)})
            scanned += ivf.last_scanned
        ms = (time.perf_counter() - t) * 1000 / n_queries
        rec = float(np.mean([len(a & b) / 10 for a, b in zip(res, truth)]))
        rows.append(
            {
                "nprobe": nprobe,
                "recall@10": round(rec, 4),
                "latency_ms": round(ms, 4),
                "scanned_frac": round(scanned / (n * n_queries), 4),
            }
        )
    return {"n_vectors": n, "dim": dim, "n_lists": n_lists, "build_ms": round(build_ms, 1), "sweep": rows}


def _filtered_eval(searcher: HybridSearcher, queries: list[Query], config: SearchConfig) -> dict:
    """Same queries, with a pre-retrieval filter on the target's domain."""
    domain_of = {d.id: d.metadata["domain"] for d in searcher.docs}
    filtered: list[Query] = []
    for q in queries:
        target = next(d for d, g in q.qrels.items() if g == 2)
        filtered.append(Query(q.id, q.text, q.qrels, q.qtype, {"domain": domain_of[target]}))
    unf = evaluate(searcher, queries, config)
    fil = evaluate(searcher, filtered, config)
    violations = 0
    for q in filtered[:60]:
        for h in searcher.search(q.text, 10, config, q.filter).hits:
            violations += searcher.docs[h.index].metadata["domain"] != q.filter["domain"]
    return {
        "config": config.label,
        "mrr_unfiltered": unf["mrr"],
        "mrr_domain_filtered": fil["mrr"],
        "latency_ms_p50_unfiltered": unf["latency_ms_p50"],
        "latency_ms_p50_filtered": fil["latency_ms_p50"],
        "filter_violations": violations,
    }


def benchmark() -> dict:
    t_all = time.perf_counter()
    docs, queries = load_benchmark()
    train, test = split_queries(queries)
    searcher = HybridSearcher(docs)
    t = time.perf_counter()
    rr = train_reranker(searcher, train)
    train_ms = (time.perf_counter() - t) * 1000

    results = sweep(searcher, test, default_sweep_configs())
    frontier = pareto_frontier(results, "latency_ms_p50", "ndcg@10")
    frontier_mrr = pareto_frontier(results, "latency_ms_p50", "mrr")
    best = max(results, key=lambda r: (r["mrr"], -r["latency_ms_p50"]))

    def brief(r: dict) -> dict:
        return {k: r[k] for k in ("config", "recall@1", "recall@10", "mrr", "ndcg@10", "latency_ms_p50", "latency_ms_p95")}

    # LLM rerank cost model: prompt size for a typical depth-20 rerank call.
    llm_rr = LLMReranker(MockLLM(default='{"scores": []}'), [d.full_text for d in docs])
    cfg = SearchConfig(mode="hybrid", candidate_k=50)
    tok = []
    for q in test[:20]:
        llm_rr.rerank(q.text, searcher.embed_query(q.text), searcher.candidates(q.text, cfg)[:20])
        tok.append(llm_rr.last_prompt_tokens)

    report = {
        "corpus": {
            "n_docs": len(docs),
            "n_queries": len(queries),
            "n_train": len(train),
            "n_test": len(test),
            "vocabulary": searcher.bm25.vocabulary_size,
            "index_build_ms": round(searcher.index_build_ms, 1),
        },
        "reranker": {
            "train_ms": round(train_ms, 1),
            "newton_iterations": rr.model.n_iter_,
            "standardized_weights": rr.feature_weights(),
        },
        "sweep": [brief(r) | {"mrr_by_type": {t: v["mrr"] for t, v in r["by_type"].items()}} for r in results],
        "pareto_frontier_ndcg": [brief(r) for r in frontier],
        "pareto_frontier_mrr": [brief(r) for r in frontier_mrr],
        "best_by_mrr": brief(best),
        "filtering": _filtered_eval(searcher, test, SearchConfig(mode="hybrid", fusion="weighted")),
        "ivf_scaling": _ivf_scaling(),
        "llm_rerank_prompt_tokens_depth20_mean": round(float(np.mean(tok)), 1),
    }
    report["total_runtime_s"] = round(time.perf_counter() - t_all, 2)
    return report


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
