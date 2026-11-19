"""End-to-end Context Assembler demo (offline): AIKIT_OFFLINE=1 python projects/p01_context_assembler/demo.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent.parent / "common")]

from aikit import count_tokens, get_llm  # noqa: E402
from context_assembler import (  # noqa: E402
    ContextAssembler, LLMSummarizer, count_payload_tokens, history_block, memory_block, query_block,
    retrieval_block, system_block, tool_results_block, tools_block,
)
from context_assembler.bench import SYSTEM, benchmark, make_case  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


TOOLS = [
    {"name": "lookup_order", "description": "Look up an order's status, items and shipping by order id",
     "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "issue_refund", "description": "Issue a refund for a returned order",
     "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
                      "required": ["order_id"]}},
    {"name": "get_weather", "description": "Get the weather forecast for a city",
     "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}},
    {"name": "translate_text", "description": "Translate text between languages",
     "input_schema": {"type": "object", "properties": {"text": {"type": "string"}, "target": {"type": "string"}}}},
    {"name": "create_calendar_event", "description": "Create a calendar event with attendees",
     "input_schema": {"type": "object", "properties": {"title": {"type": "string"}}}},
]


def build_blocks():
    case = make_case(seed=3)
    blocks = [
        system_block(SYSTEM),
        history_block(case["hist"]),
        retrieval_block(case["docs"], top_k=6),
        memory_block([
            {"text": "Customer is a premium member since 2019", "importance": 0.9},
            {"text": "Customer prefers concise answers", "importance": 0.6},
            {"text": "Customer once asked about gift wrapping", "importance": 0.1},
        ], top_k=2),
        tools_block(TOOLS, pinned=["lookup_order"], top_k=3),
        tool_results_block([{"name": "lookup_order",
                             "content": "order_id=A-1029 status=DELIVERED_LATE " + "event=scan location=hub " * 300
                                        + "final=delivered 2026-11-12 (6 days late)"}]),
        query_block(case["query"]),
    ]
    return case, blocks


def main() -> None:
    case, blocks = build_blocks()
    demand = sum(count_tokens(it.text) for b in blocks for it in b.items)
    print(f"Raw context available: {len(blocks)} sources, ~{demand:,} tokens of content")

    rule("1. Assemble into an 8k window (2k reserved for the answer)")
    asm = ContextAssembler(window=8192, reserve_output=2048)
    out = asm.assemble(blocks)
    print(out.report.render_text())

    rule("2. The guarantee: measured payload vs. limit")
    measured = count_payload_tokens(out.payload, count_tokens)
    print(f"payload tokens = {measured}  <=  window - reserve = {asm.input_limit}:  {measured <= asm.input_limit}")
    print(f"needle document kept:     {'needle' in out.payload['messages'][-1]['content']}")
    print(f"buried fact {case['code']} kept: {case['code'] in json.dumps(out.payload)}  (via history summary)")

    rule("3. Messages-API payload (abridged)")
    p = out.payload
    print("system blocks :", [(b["text"][:60] + "...", "cache_control" in b) for b in p["system"]])
    print("tools         :", [t["name"] for t in p["tools"]], "(pinned lookup_order + most relevant)")
    print("messages      :", len(p["messages"]), "alternating turns, first role =", p["messages"][0]["role"])
    print("summary turn  :", p["messages"][0]["content"][:300], "...")
    final = p["messages"][-1]["content"]
    print("final turn    : docs order ->", [seg.split('"')[1] for seg in final.split("<document id=")[1:]],
          "(best first, 2nd-best last: lost-in-the-middle mitigation)")
    print("final turn    : ends with ->", final[-100:].replace("\n", " "))

    rule("4. Same sources, three window sizes")
    for window in (2048, 8192, 32768):
        r = ContextAssembler(window=window, reserve_output=1024).assemble(blocks).report
        kinds = ", ".join(f"{k}={v}" for k, v in r.by_kind.items())
        print(f"window={window:>6}: payload={r.payload_tokens:>6} util={r.utilization:6.1%}  [{kinds}]")

    rule("5. Pluggable LLM summarizer (Claude if ANTHROPIC_API_KEY is set, else offline extractive)")
    llm = get_llm()
    if type(llm).__name__ == "ClaudeLLM":
        out2 = ContextAssembler(window=8192, reserve_output=2048, summarizer=LLMSummarizer(llm)).assemble(blocks)
        print(out2.payload["messages"][0]["content"][:400])
        system, prompt = out2.to_prompt()
        print("\nClaude answer:", llm.complete(prompt, system=system, max_tokens=400).text[:500])
    else:
        print("offline mode: the extractive summarizer above was used; the LLM path is covered by tests with MockLLM")

    rule("6. Benchmark")
    b = benchmark()
    o = b["overflow_property"]
    print(f"overflow property: {o['overflows']} overflows in {o['cases']} randomized cases "
          f"(median utilization when constrained {o['utilization_when_constrained_median']:.1%})")
    print("retention (needle doc / buried fact / query / system / all four), 30 long-conversation cases:")
    for name, r in b["retention_vs_naive"]["retention_rates"].items():
        print(f"   {name:31} {r['needle_doc']:.2f} / {r['buried_fact']:.2f} / {r['query']:.2f} / "
              f"{r['system_prompt']:.2f} / {r['all_four']:.2f}")
    for w, l in b["latency"].items():
        if isinstance(l, dict):
            print(f"latency {w}: p50 {l['p50_ms']} ms, p95 {l['p95_ms']} ms")
    c = b["cost"]
    print(f"cost per 1k requests: ${c['assembled_usd_per_1k_requests']} vs ${c['full_context_usd_per_1k_requests']} "
          f"for the full context ({c['token_reduction']:.0%} fewer input tokens)")


if __name__ == "__main__":
    main()
