"""Benchmark: overflow safety, latency, information retention vs. naive truncation, cost.

Run: ``python -m context_assembler.bench`` from the project directory.
All inputs are generated from fixed seeds, so accuracy numbers are deterministic.
"""
from __future__ import annotations

import json
import random
import statistics
import time
from typing import Any

from aikit import HashingEmbedder, count_tokens, truncate_to_tokens

from .assembler import ContextAssembler
from .blocks import history_block, memory_block, query_block, retrieval_block, system_block, tool_results_block, tools_block
from .models import BlockKind, ContextBlock, ContextItem, ContextOverflowError, Placement, Strategy
from .render import count_payload_tokens

PRICE_IN_PER_MTOK = 4.0  # claude-opus-5-5 input price (USD / million tokens)

TOPICS = ["shipping delays", "warehouse staffing", "carrier contracts", "packaging costs", "fleet maintenance",
          "customs paperwork", "inventory audits", "holiday peak planning", "returns logistics", "driver safety"]
SYSTEM = ("You are Acme's senior support agent. Answer using the provided documents and conversation. "
          "Cite document ids. If information is missing, say so. Never invent order numbers.")


def _doc(rng: random.Random, i: int) -> str:
    t = rng.choice(TOPICS)
    return " ".join(
        f"Section {i} on {t}: teams reviewed {t} metrics, weekly reports and escalation paths for region "
        f"{rng.randint(1, 9)}." for _ in range(rng.randint(3, 7))
    )


def make_case(seed: int, n_docs: int = 60, n_turns: int = 80) -> dict[str, Any]:
    """A long support conversation with one buried fact and one needle document."""
    rng = random.Random(seed)
    code = f"RMA-{rng.randint(1000, 9999)}"
    fact_turn = rng.randint(2, 12) * 2  # an early user turn
    hist = []
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "assistant"
        if i == fact_turn:
            text = f"For the record, my return authorization code is {code} and the parcel weighs 4 kg."
        else:
            text = (f"Turn {i}: let's talk about {rng.choice(TOPICS)} for a moment, it affects the team and "
                    f"the weekly schedule quite a lot.")
        hist.append({"role": role, "content": text})
    docs = [{"id": f"doc{i}", "text": _doc(rng, i), "score": round(rng.uniform(0.3, 0.6), 3)} for i in range(n_docs)]
    needle_pos = rng.randrange(n_docs)
    docs[needle_pos] = {"id": "needle", "score": round(rng.uniform(0.3, 0.6), 3),
                        "text": "Refund policy: a return authorization code entitles the customer to a full refund "
                                "within 30 days, shipped back at Acme's expense."}
    query = "I have a return authorization code. Am I entitled to a full refund, and what's my code again?"
    return {"hist": hist, "docs": docs, "query": query, "code": code}


def assembler_blocks(case: dict[str, Any]) -> list[ContextBlock]:
    return [
        system_block(SYSTEM),
        history_block(case["hist"]),
        retrieval_block(case["docs"], top_k=8),
        memory_block([{"text": "Customer is a premium member since 2019", "importance": 0.8}]),
        query_block(case["query"]),
    ]


def naive_text(case: dict[str, Any]) -> str:
    """Everything concatenated in the conventional order: system, history, docs, query."""
    parts = [SYSTEM] + [f"{m['role']}: {m['content']}" for m in case["hist"]]
    parts += [f"[{d['id']}] {d['text']}" for d in case["docs"]] + [case["query"]]
    return "\n".join(parts)


def keep_tail(text: str, budget: int) -> str:
    words = text.split(" ")
    lo, hi = 0, len(words)
    while lo < hi:  # smallest start index whose suffix fits
        mid = (lo + hi) // 2
        if count_tokens(" ".join(words[mid:])) <= budget:
            hi = mid
        else:
            lo = mid + 1
    return " ".join(words[lo:])


def _retention(text: str, case: dict[str, Any]) -> dict[str, bool]:
    return {
        "needle_doc": "Refund policy: a return authorization" in text,
        "buried_fact": case["code"] in text,
        "query": case["query"] in text,
        "system_prompt": SYSTEM in text,
    }


def _payload_text(payload: dict[str, Any]) -> str:
    sys_ = payload.get("system", "")
    sys_text = sys_ if isinstance(sys_, str) else "\n".join(b["text"] for b in sys_)
    return sys_text + "\n" + "\n".join(m["content"] for m in payload["messages"])


def retention_benchmark(n_cases: int = 30, window: int = 4096, reserve: int = 1024) -> dict[str, Any]:
    limit = window - reserve
    systems = {"assembler": [], "assembler_drop_oldest_history": [], "naive_keep_head": [], "naive_keep_tail": []}
    full_tokens, sent_tokens = [], []
    for seed in range(n_cases):
        case = make_case(seed)
        out = ContextAssembler(window=window, reserve_output=reserve).assemble(assembler_blocks(case))
        assert out.report.payload_tokens <= limit
        systems["assembler"].append(_retention(_payload_text(out.payload), case))
        ablation = [b.model_copy(update={"strategy": Strategy.DROP_OLDEST}) if b.kind == BlockKind.HISTORY else b
                    for b in assembler_blocks(case)]
        abl = ContextAssembler(window=window, reserve_output=reserve).assemble(ablation)
        systems["assembler_drop_oldest_history"].append(_retention(_payload_text(abl.payload), case))
        text = naive_text(case)
        full_tokens.append(count_tokens(text))
        sent_tokens.append(out.report.payload_tokens)
        systems["naive_keep_head"].append(_retention(truncate_to_tokens(text, limit), case))
        systems["naive_keep_tail"].append(_retention(keep_tail(text, limit), case))
    rates = {
        name: {k: round(sum(r[k] for r in rows) / len(rows), 3) for k in rows[0]} for name, rows in systems.items()
    }
    for name, r in rates.items():
        r["all_four"] = round(sum(all(x.values()) for x in systems[name]) / n_cases, 3)
    return {
        "cases": n_cases, "window": window, "reserve_output": reserve,
        "mean_full_context_tokens": round(statistics.mean(full_tokens), 1),
        "mean_sent_tokens": round(statistics.mean(sent_tokens), 1),
        "retention_rates": rates,
    }


def _random_blocks(rng: random.Random) -> list[ContextBlock]:
    words = "alpha beta refund policy order kafka latency internationalization 42 $5 , . ( ) <x> ü".split()
    blocks = []
    for bi in range(rng.randint(1, 8)):
        kind = rng.choice(list(BlockKind))
        items = []
        for ii in range(rng.randint(0, 15)):
            kw: dict[str, Any] = dict(id=f"b{bi}i{ii}", text=" ".join(rng.choice(words) for _ in range(rng.randint(0, 200))),
                                      pinned=rng.random() < 0.1, score=rng.random())
            if kind == BlockKind.HISTORY:
                kw["role"] = rng.choice(["user", "assistant"])
            if kind == BlockKind.TOOL_DEFS:
                kw["tool"] = {"name": f"t{bi}_{ii}", "description": kw["text"][:200], "input_schema": {"type": "object"}}
            items.append(ContextItem(**kw))
        blocks.append(ContextBlock(name=f"b{bi}", kind=kind, items=items, priority=rng.randint(0, 100),
                                   required=rng.random() < 0.3, strategy=rng.choice(list(Strategy)),
                                   placement=rng.choice(list(Placement)), weight=rng.choice([0.5, 1, 3]),
                                   min_tokens=rng.choice([None, 0, 50]), top_k=rng.choice([None, 3])))
    return blocks


def overflow_benchmark(n_cases: int = 250) -> dict[str, Any]:
    rng = random.Random(1234)
    embedder = HashingEmbedder()
    overflows = raised = repairs = 0
    utils = []
    for _ in range(n_cases):
        window = rng.choice([64, 256, 1024, 4096, 16_384])
        asm = ContextAssembler(window=window, reserve_output=rng.randint(0, window // 3), embedder=embedder)
        try:
            out = asm.assemble(_random_blocks(rng), query="what is the refund policy?")
        except ContextOverflowError:
            raised += 1
            continue
        measured = count_payload_tokens(out.payload, count_tokens)
        overflows += measured > asm.input_limit
        repairs += out.report.repair_rounds
        demand = sum(b.demand for b in out.report.blocks) + 7
        if demand > asm.input_limit:  # only meaningful when content had to be cut
            utils.append(measured / asm.input_limit)
    return {
        "cases": n_cases, "overflows": overflows, "overflow_errors_raised": raised, "repair_rounds": repairs,
        "constrained_cases": len(utils),
        "utilization_when_constrained_mean": round(statistics.mean(utils), 4),
        "utilization_when_constrained_median": round(statistics.median(utils), 4),
    }


def latency_benchmark(reps: int = 15) -> dict[str, Any]:
    out: dict[str, Any] = {}
    case = make_case(7, n_docs=200, n_turns=300)
    tools = [{"name": f"tool_{i}", "description": f"Tool {i} for {TOPICS[i % len(TOPICS)]}",
              "input_schema": {"type": "object", "properties": {"arg": {"type": "string"}}}} for i in range(30)]
    blocks = assembler_blocks(case) + [
        tools_block(tools, top_k=8),
        tool_results_block([{"name": "tool_1", "content": "row value " * 2000}]),
    ]
    demand = None
    for window in (8_192, 32_768, 200_000):
        asm = ContextAssembler(window=window, reserve_output=2048)
        asm.assemble(blocks)  # warm the embedding cache (steady-state service behaviour)
        times = []
        for _ in range(reps):
            t = time.perf_counter()
            res = asm.assemble(blocks)
            times.append((time.perf_counter() - t) * 1000)
        times.sort()
        demand = sum(b.demand for b in res.report.blocks)
        out[f"window_{window}"] = {
            "p50_ms": round(times[len(times) // 2], 2), "p95_ms": round(times[int(len(times) * 0.95) - 1], 2),
            "payload_tokens": res.report.payload_tokens, "utilization": res.report.utilization,
        }
    out["input_demand_tokens"] = demand
    return out


def benchmark() -> dict[str, Any]:
    t0 = time.perf_counter()
    retention = retention_benchmark()
    overflow = overflow_benchmark()
    latency = latency_benchmark()
    full = retention["mean_full_context_tokens"]
    sent = retention["mean_sent_tokens"]
    return {
        "overflow_property": overflow,
        "retention_vs_naive": retention,
        "latency": latency,
        "cost": {
            "input_price_usd_per_mtok": PRICE_IN_PER_MTOK,
            "full_context_usd_per_1k_requests": round(full * 1000 * PRICE_IN_PER_MTOK / 1e6, 3),
            "assembled_usd_per_1k_requests": round(sent * 1000 * PRICE_IN_PER_MTOK / 1e6, 3),
            "token_reduction": round(1 - sent / full, 4),
        },
        "total_runtime_s": round(time.perf_counter() - t0, 2),
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
