"""End-to-end demo of the hybrid retrieval stack (runs fully offline).

    AIKIT_OFFLINE=1 .venv/bin/python projects/p02_hybrid_retrieval/demo.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parents[1] / "common")]

from aikit import ClaudeLLM, count_tokens, get_llm  # noqa: E402

from hybrid_retrieval import (  # noqa: E402
    Eq,
    HybridSearcher,
    In,
    LLMReranker,
    Range,
    SearchConfig,
    evaluate,
    load_benchmark,
    pareto_frontier,
    split_queries,
    sweep,
    train_reranker,
)

CONFIGS = {
    "sparse (BM25)": SearchConfig(mode="sparse"),
    "dense (flat)": SearchConfig(mode="dense"),
    "hybrid (weighted)": SearchConfig(mode="hybrid", fusion="weighted"),
    "hybrid + rerank": SearchConfig(mode="hybrid", fusion="weighted", rerank=True, rerank_depth=40),
}


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    docs, queries = load_benchmark()
    train, test = split_queries(queries)
    searcher = HybridSearcher(docs)
    rule("1. Index")
    print(f"{len(docs)} docs | BM25 vocabulary {searcher.bm25.vocabulary_size} terms | "
          f"dense dim {searcher.vectors.shape[1]} | IVF lists {searcher.ivf.n_lists} | "
          f"built in {searcher.index_build_ms:.0f} ms")
    print(f"{len(queries)} labeled queries -> {len(train)} train (reranker) / {len(test)} test")

    rr = train_reranker(searcher, train)
    rule("2. Learned cross-encoder-style reranker (numpy logistic regression)")
    for name, w in sorted(rr.feature_weights().items(), key=lambda kv: -abs(kv[1])):
        print(f"  {name:16s} {w:+.3f}")

    rule("3. One query per family: top-1 per config (target marked with *)")
    examples = {}
    for q in test:
        examples.setdefault(q.qtype, q)
    for qtype, q in sorted(examples.items()):
        target = next(d for d, g in q.qrels.items() if g == 2)
        print(f"\n[{qtype}] {q.text!r}   target={target}")
        for label, cfg in CONFIGS.items():
            resp = searcher.search(q.text, 5, cfg)
            ids = resp.doc_ids
            rank = ids.index(target) + 1 if target in ids else None
            mark = "*" if ids and ids[0] == target else " "
            print(f"  {label:18s} {mark} top1={ids[0] if ids else '-':34s} target_rank={rank}  "
                  f"({resp.timings_ms['total']:.2f} ms)")

    rule("4. Accuracy by query family (MRR on held-out test queries)")
    print(f"  {'config':18s} {'overall':>8s} {'keyword':>8s} {'lexical':>8s} {'mixed':>8s} {'paraphr':>8s} {'p50 ms':>8s}")
    for label, cfg in CONFIGS.items():
        r = evaluate(searcher, test, cfg)
        bt = r["by_type"]
        print(f"  {label:18s} {r['mrr']:8.3f} {bt['keyword']['mrr']:8.3f} {bt['lexical']['mrr']:8.3f} "
              f"{bt['mixed']['mrr']:8.3f} {bt['paraphrase']['mrr']:8.3f} {r['latency_ms_p50']:8.3f}")

    rule("5. Pre-retrieval metadata filtering")
    flt = (Eq("domain", "cloud_database") & Range("year", gte=2021)) | (In("source", ["forum"]) & Eq("aspect", "errors"))
    print("filter:", flt.to_dict())
    resp = searcher.search("database crash with error code", 5, CONFIGS["hybrid (weighted)"], filter=flt)
    for h in resp.hits:
        m = searcher.docs[h.index].metadata
        print(f"  {h.doc_id:32s} domain={m['domain']:15s} year={m['year']} source={m['source']}  score={h.score:.3f}")
    json_flt = {"$and": [{"domain": "telehealth"}, {"aspect": {"$in": ["refund", "pricing"]}}]}
    resp = searcher.search("how expensive", 3, CONFIGS["hybrid (weighted)"], filter=json_flt)
    print("JSON DSL", json_flt, "->", resp.doc_ids)

    rule("6. Latency vs. accuracy sweep -> Pareto frontier (nDCG@10 vs p50 latency)")
    results = sweep(searcher, test)
    frontier = {r["config"] for r in pareto_frontier(results)}
    for r in sorted(results, key=lambda r: r["latency_ms_p50"]):
        star = "<- pareto" if r["config"] in frontier else ""
        print(f"  {r['config']:32s} nDCG@10={r['ndcg@10']:.3f} MRR={r['mrr']:.3f} "
              f"R@10={r['recall@10']:.3f} p50={r['latency_ms_p50']:.3f}ms {star}")

    rule("7. LLM reranker (pluggable via aikit.LLM)")
    llm = get_llm()
    llm_rr = LLMReranker(llm, [d.full_text for d in docs])
    q = examples["paraphrase"]
    hits = searcher.candidates(q.text, SearchConfig(fusion="weighted"))[:10]
    prompt_tokens = count_tokens(llm_rr.build_prompt(q.text, hits))
    if isinstance(llm, ClaudeLLM):
        out = llm_rr.rerank(q.text, searcher.embed_query(q.text), hits)
        print(f"Claude reranked {q.text!r}: {[h.doc_id for h in out[:3]]}")
    else:
        print(f"Offline (AIKIT_OFFLINE={os.environ.get('AIKIT_OFFLINE', '')!r} / no API key): "
              f"skipping live LLM rerank; a depth-10 rerank prompt is ~{prompt_tokens} tokens. "
              "The learned reranker above is the offline default.")


if __name__ == "__main__":
    main()
