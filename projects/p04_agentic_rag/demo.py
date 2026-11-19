"""End-to-end demo of the self-correcting agentic RAG pipeline (runs fully offline).

    AIKIT_OFFLINE=1 .venv/bin/python projects/p04_agentic_rag/demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent.parent / "common")]

from agentic_rag import AgenticRAG, ExtractiveGenerator, HallucinatingGenerator  # noqa: E402
from agentic_rag.bench import benchmark  # noqa: E402


def show(title: str, rag: AgenticRAG, question: str, trace: bool = False) -> None:
    r = rag.run(question)
    print(f"\n=== {title} ===")
    print(f"Q: {question}")
    print(f"A: {r.answer}")
    print(f"   abstained={r.abstained} confidence={r.confidence:.3f} signals={r.signals}")
    for chk in r.flagged:
        print(f"   FLAGGED [{chk.verdict.label}] {chk.claim.text!r}  <- {chk.verdict.reason}")
    if trace:
        print("   trace:")
        print(r.trace.pretty())


def main() -> None:
    rag = AgenticRAG()
    show("Answerable (synonym match: vacation ~ paid time off)", rag,
         "How many vacation days do Lumora employees get?")
    show("Multi-part question (decomposed into sub-questions)", rag,
         "What is the Beacon battery life and how much does it weigh?", trace=True)
    show("Unanswerable near-miss (should abstain, explaining what is missing)", rag,
         "What is the battery life of the Lumora Pulse?", trace=True)
    show("Unanswerable (salary not in the handbook)", rag, "What is the salary of the Lumora CEO?")

    liar = AgenticRAG(generator=HallucinatingGenerator(
        ExtractiveGenerator(rag.stats), [d.doc_id for d in rag.docs], seed=3))
    show("Adversarial: hallucinating generator injected (detector + repair)", liar,
         "When was Lumora founded and who is its CEO?", trace=True)

    print("\n=== Benchmark ===")
    print(json.dumps(benchmark(), indent=2))


if __name__ == "__main__":
    main()
