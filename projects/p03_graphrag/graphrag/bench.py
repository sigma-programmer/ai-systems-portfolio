"""Benchmark: GraphRAG multi-hop accuracy vs. naive vector RAG, plus pipeline costs.

Run: ``python -m graphrag.bench`` (from the project dir, with ``common`` on the path).
"""
from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from typing import Any

from aikit import count_tokens

from .baseline import VectorRAG
from .corpus import DOCUMENTS, GLOBAL_QUESTIONS, GOLD_TRIPLES, QUESTIONS
from .pipeline import GraphRAG

#: Illustrative Claude-class pricing (USD per million tokens) for the cost estimate.
PRICE_IN_PER_MTOK = 5.0
PRICE_OUT_PER_MTOK = 25.0


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 3)


def benchmark() -> dict[str, Any]:
    """Index the bundled corpus, answer every QA item both ways, and report metrics."""
    t0 = time.perf_counter()
    rag = GraphRAG().index(DOCUMENTS)
    index_s = time.perf_counter() - t0

    triples = rag.kg.triples()
    tp = len(triples & GOLD_TRIPLES)
    extraction = {
        "precision": round(tp / max(1, len(triples)), 4),
        "recall": round(tp / len(GOLD_TRIPLES), 4),
        "typed_triples": len(triples),
        "gold_triples": len(GOLD_TRIPLES),
    }

    systems: dict[str, Any] = {}
    per_hop: dict[str, dict[int, list[bool]]] = {}
    for k in (3, 5):
        vec = VectorRAG(rag.chunks, k=k)
        name = f"vector_rag_top{k}"
        lat, hits, ctx = [], defaultdict(list), []
        support_recall = []
        for q in QUESTIONS:
            t = time.perf_counter()
            ans = vec.answer(q.question)
            lat.append(time.perf_counter() - t)
            hits[q.hops].append(ans.answer == q.answer)
            ctx.append(sum(count_tokens(c.text) for c, _ in vec.retrieve(q.question)))
            got = {cid.split("#")[0] for cid in ans.retrieved}
            support_recall.append(len(got & set(q.support)) / len(q.support))
        per_hop[name] = hits
        systems[name] = {
            "p50_latency_ms": _ms(statistics.median(lat)),
            "mean_context_tokens": round(statistics.mean(ctx), 1),
            "support_doc_recall": round(statistics.mean(support_recall), 4),
        }

    lat, hits, ctx, support_recall, path_lens = [], defaultdict(list), [], [], []
    for q in QUESTIONS:
        t = time.perf_counter()
        ans = rag.local_search(q.question, max_hops=3)
        lat.append(time.perf_counter() - t)
        hits[q.hops].append(ans.answer == q.answer)
        ctx.append(sum(count_tokens(rag.kg.chunk_text[c]) for c in ans.supporting_chunks))
        got = {c.split("#")[0] for c in ans.supporting_chunks}
        support_recall.append(len(got & set(q.support)) / len(q.support))
        if ans.paths:
            path_lens.append(len(ans.paths[0]))
    per_hop["graphrag_local"] = hits
    systems["graphrag_local"] = {
        "p50_latency_ms": _ms(statistics.median(lat)),
        "mean_context_tokens": round(statistics.mean(ctx), 1),
        "support_doc_recall": round(statistics.mean(support_recall), 4),
        "mean_path_length": round(statistics.mean(path_lens), 3),
    }

    for name, hits in per_hop.items():
        all_hits = [h for v in hits.values() for h in v]
        multi = [h for hop, v in hits.items() if hop >= 2 for h in v]
        systems[name]["accuracy_overall"] = round(sum(all_hits) / len(all_hits), 4)
        systems[name]["accuracy_multihop"] = round(sum(multi) / len(multi), 4)
        systems[name]["accuracy_by_hops"] = {str(h): round(sum(v) / len(v), 4) for h, v in sorted(hits.items())}

    g_lat = []
    g_points = []
    for q in GLOBAL_QUESTIONS:
        t = time.perf_counter()
        ans = rag.global_search(q)
        g_lat.append(time.perf_counter() - t)
        g_points.append(len(ans.points))
    n_comm0 = len([c for c in rag.communities if c.level == 0])
    summary_tokens = sum(count_tokens(c.summary) for c in rag.communities if c.level == 0)

    # LLM-mode cost estimate: one extraction call per chunk (prompt ≈ 230 tokens of
    # instructions + chunk; reply ≈ 1.5x chunk) and one map call per community.
    chunk_tokens = sum(c.n_tokens for c in rag.chunks)
    llm_in = len(rag.chunks) * 230 + chunk_tokens
    llm_out = int(chunk_tokens * 1.5)
    index_cost = (llm_in * PRICE_IN_PER_MTOK + llm_out * PRICE_OUT_PER_MTOK) / 1e6

    return {
        "corpus": {"documents": len(DOCUMENTS), "chunks": len(rag.chunks), "chunk_tokens": chunk_tokens,
                   "questions": len(QUESTIONS), "multihop_questions": sum(q.hops >= 2 for q in QUESTIONS)},
        "graph": rag.stats(),
        "index_latency_ms": _ms(index_s),
        "index_stage_ms": rag.timings_ms,
        "extraction_vs_gold": extraction,
        "systems": systems,
        "global_search": {
            "questions": len(GLOBAL_QUESTIONS),
            "p50_latency_ms": _ms(statistics.median(g_lat)),
            "mean_points": round(statistics.mean(g_points), 2),
            "level0_communities": n_comm0,
            "level0_summary_tokens": summary_tokens,
        },
        "llm_mode_estimate": {
            "extraction_input_tokens": llm_in,
            "extraction_output_tokens": llm_out,
            "index_cost_usd": round(index_cost, 4),
            "global_query_map_calls": n_comm0,
        },
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
