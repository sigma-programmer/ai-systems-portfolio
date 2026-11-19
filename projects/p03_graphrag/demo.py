"""End-to-end GraphRAG demo (offline): AIKIT_OFFLINE=1 python projects/p03_graphrag/demo.py"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent.parent / "common")]

from aikit import get_llm  # noqa: E402
from graphrag import DOCUMENTS, GLOBAL_QUESTIONS, QUESTIONS, GraphRAG, VectorRAG  # noqa: E402
from graphrag.bench import benchmark  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    llm = get_llm()
    live = type(llm).__name__ == "ClaudeLLM"
    rag = GraphRAG(llm=llm if live else None).index(DOCUMENTS)
    print(f"LLM mode: {'Claude' if live else 'offline heuristics (no LLM)'}")

    rule("1. Unstructured text -> knowledge graph")
    print(json.dumps(rag.stats(), indent=2))
    print("timings (ms):", rag.timings_ms)
    print("\nSample typed triples:")
    for t in sorted(rag.kg.triples())[:8]:
        print("  ", t)
    print("\nEntity resolution (surface -> canonical):")
    for surface in ("Quilon Systems", "NF", "Okafor", "Zentrix", "Zentrix Labs Inc."):
        node = rag.kg.find(surface)
        print(f"   {surface!r:22} -> {rag.kg.name(node) if node else None}")

    rule("2. Communities (Louvain, hierarchical) and extractive summaries")
    for c in [c for c in rag.communities if c.level == 0][:4]:
        print(f"[{c.id}] {c.summary[:240]}{'...' if len(c.summary) > 240 else ''}\n")

    rule("3. Local search: multi-hop traversal with path explanations")
    vec = VectorRAG(rag.chunks, k=3)
    for q in [q for q in QUESTIONS if q.hops >= 2][::4]:
        ans = rag.local_search(q.question, max_hops=3)
        v = vec.answer(q.question)
        print(f"Q ({q.hops}-hop): {q.question}")
        print(f"   logical form : {ans.logical_form}")
        print(f"   path         : {ans.explanation}")
        print(f"   sources      : {ans.supporting_chunks}")
        print(f"   GraphRAG     : {ans.answer!s:20} {'OK' if ans.answer == q.answer else 'WRONG'}")
        print(f"   Vector RAG   : {v.answer!s:20} {'OK' if v.answer == q.answer else 'WRONG'} (gold: {q.answer})\n")

    rule("4. Global search: map-reduce over community summaries")
    for q in GLOBAL_QUESTIONS:
        ans = rag.global_search(q)
        print(f"Q: {q}\n{ans.answer}\n")

    rule("5. Export")
    out = Path(tempfile.gettempdir()) / "graphrag_kg.json"
    rag.save(out)
    print(f"graph written to {out} ({out.stat().st_size} bytes)")

    rule("6. Benchmark: multi-hop QA accuracy")
    b = benchmark()
    print(f"{'system':18} {'overall':>8} {'multihop':>9} {'1-hop':>6} {'2-hop':>6} {'3-hop':>6} {'ctx tok':>8} {'p50 ms':>7}")
    for name, s in b["systems"].items():
        h = s["accuracy_by_hops"]
        print(f"{name:18} {s['accuracy_overall']:8.3f} {s['accuracy_multihop']:9.3f} {h['1']:6.2f} {h['2']:6.2f} "
              f"{h['3']:6.2f} {s['mean_context_tokens']:8.1f} {s['p50_latency_ms']:7.3f}")
    print("extraction vs gold:", b["extraction_vs_gold"])


if __name__ == "__main__":
    main()
