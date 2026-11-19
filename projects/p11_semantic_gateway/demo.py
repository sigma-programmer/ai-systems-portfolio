"""End-to-end demo of the semantic cache + routing gateway (fully offline, simulated backends).

    AIKIT_OFFLINE=1 .venv/bin/python projects/p11_semantic_gateway/demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent.parent / "common")]

from semantic_gateway import (CostTracker, Gateway, GatewayRequest, ManualClock,  # noqa: E402
                              SemanticCache, TenantBudget, default_backends)
from semantic_gateway.bench import benchmark  # noqa: E402


def show(gw: Gateway, prompt: str, tenant: str = "acme") -> None:
    r = gw.handle(GatewayRequest(prompt, tenant))
    c = r.complexity
    what = ("CACHE HIT" + (f" sim={r.cache_similarity:.2f}" if r.cache_similarity else "")
            if r.cached else f"{r.model}")
    print(f"[{tenant:7}] {prompt[:62]!r:66} -> {what:24} tier={r.tier or '-':8} "
          f"score={c.score if c else '-'} cost=${r.cost_usd:.6f} "
          f"{'DEGRADED ' if r.degraded else ''}{r.error or ''} decision={r.decision}")


def main() -> None:
    clock = ManualClock()
    backends = default_backends(seed=1)
    tracker = CostTracker(budgets={"initech": TenantBudget(daily_cap_usd=0.0011)}, clock=clock)
    gw = Gateway(backends, cache=SemanticCache(clock=clock), tracker=tracker, clock=clock)

    print("=== Complexity routing ===")
    for p in ("What is the capital of France?",
              "Summarize the main causes of World War I in a paragraph.",
              "Prove that the square root of 2 is irrational, step by step."):
        show(gw, p)

    print("\n=== Semantic cache: paraphrases hit, look-alikes are guarded ===")
    show(gw, "Which city is the capital of France?")
    show(gw, "What is the capital of Peru?")             # slot substitution -> miss
    show(gw, "Convert 5 km to miles.")
    show(gw, "Convert 8 km to miles.")                    # number mismatch -> miss
    show(gw, "Which city is the capital of France?", tenant="globex")  # tenant isolation

    print("\n=== Kill-switch: initech has a $0.0011/day cap ===")
    for p in ("Design a URL shortener architecture and analyze the trade-offs step by step.",
              "Explain recursion to a beginner with one example.",
              "Design a URL shortener architecture and analyze the trade-offs step by step.",
              "What currency is used in Kenya?"):
        clock.advance(5)
        show(gw, p, tenant="initech")

    print("\n=== Degradation: Opus outage ===")
    backends["claude-opus-5-5"].failure_rate = 1.0
    for i in range(5):
        clock.advance(2)
        r = gw.handle(GatewayRequest(f"Derive the closed-form OLS solution, case {i}, and "
                                     "justify each step rigorously."))
        print(f"  request {i}: served by {r.model}, attempts="
              f"{[(a['model'].split('-')[1], a['outcome']) for a in r.attempts]}")
    print("  opus breaker:", gw.breakers["claude-opus-5-5"].state)

    print("\n=== Benchmark (500-request synthetic workload) ===")
    print(json.dumps(benchmark(), indent=2))


if __name__ == "__main__":
    main()
