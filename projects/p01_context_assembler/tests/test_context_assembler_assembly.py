"""End-to-end assembly: payload shape, report, redistribution, and the overflow guarantee."""
from __future__ import annotations

import json
import random

import pytest

from aikit import MockLLM, count_tokens
from context_assembler import (
    BlockKind, ContextAssembler, ContextBlock, ContextItem, ContextOverflowError, LLMSummarizer, Placement,
    Strategy, count_payload_tokens, history_block, memory_block, retrieval_block, system_block,
    tool_results_block, tools_block,
)

TOOLS = [
    {"name": "search_kb", "description": "Search the knowledge base for policy documents",
     "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}},
    {"name": "issue_refund", "description": "Issue a refund for an order",
     "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}}},
]


def scenario(n_hist: int = 30, n_docs: int = 12) -> list[ContextBlock]:
    hist = [{"role": "user" if i % 2 == 0 else "assistant",
             "content": f"Turn {i}: the customer asked about order #{1000 + i} and shipping delays of {i} days."}
            for i in range(n_hist)]
    docs = [{"id": f"d{i}", "text": f"Shipping guide section {i}. Carriers, tracking numbers and delays. " * 5,
             "score": 0.5} for i in range(n_docs)]
    docs.append({"id": "refund_policy", "text": "Refunds: items may be returned within 30 days for a full refund.",
                 "score": 0.9})
    return [
        system_block("You are a support agent for Acme. Be accurate and cite documents."),
        history_block(hist),
        retrieval_block(docs, top_k=5),
        memory_block([{"text": "Customer is a premium member", "importance": 0.9},
                      {"text": "Customer prefers email", "importance": 0.3}]),
        tools_block(TOOLS),
        tool_results_block([{"name": "search_kb", "content": "hit " * 400}]),
    ]


QUERY = "Can I get a refund for order #1029 that arrived late?"


def test_everything_fits_when_window_is_large():
    out = ContextAssembler(window=100_000, reserve_output=1000).assemble(scenario(4, 3), query=QUERY)
    rep = out.report
    assert all(b.status == "full" for b in rep.blocks)
    assert rep.payload_tokens == rep.estimated_tokens
    assert not rep.warnings


def test_payload_shape_matches_messages_api():
    out = ContextAssembler(window=3000, reserve_output=500).assemble(scenario(), query=QUERY)
    p = out.payload
    assert set(p) == {"system", "messages", "tools", "max_tokens"} and p["max_tokens"] == 500
    assert p["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "<memories>" in p["system"][1]["text"]
    msgs = p["messages"]
    assert msgs[0]["role"] == "user" and msgs[-1]["role"] == "user"
    assert all(a["role"] != b["role"] for a, b in zip(msgs, msgs[1:]))
    assert msgs[-1]["content"].rstrip().endswith(QUERY)  # query is the last thing the model reads
    assert "<documents>" in msgs[-1]["content"]
    assert {t["name"] for t in p["tools"]} <= {"search_kb", "issue_refund"}
    req = out.request("claude-opus-5-5")
    assert req["model"] == "claude-opus-5-5" and json.dumps(req)
    system, prompt = out.to_prompt()
    assert "support agent" in system and prompt.endswith(QUERY)


def test_tight_window_report_explains_decisions():
    out = ContextAssembler(window=1600, reserve_output=400).assemble(scenario(), query=QUERY)
    rep = out.report
    assert rep.payload_tokens <= rep.input_limit
    assert rep.block("system").status == "full" and rep.block("query").status == "full"
    hist = rep.block("history")
    assert hist.status == "compressed" and any("summarized" in a for a in hist.actions)
    retr = rep.block("retrieval")
    assert retr.items_kept <= 5 and retr.dropped_ids
    assert "refund_policy" not in retr.dropped_ids
    assert sum(rep.by_kind.values()) + 7 == rep.estimated_tokens
    text = rep.render_text()
    assert "history" in text and "utilization" in text
    d = rep.to_dict()
    assert json.dumps(d) and d["blocks"][0]["status"]


def test_lost_in_the_middle_placement():
    docs = [{"id": f"d{i}", "text": f"refund policy clause {i} " + "filler " * i} for i in range(6)]
    blocks = [retrieval_block(docs, placement=Placement.EDGES, mmr_lambda=None)]
    out = ContextAssembler(window=100_000, reserve_output=10).assemble(blocks, query="refund policy clause")
    content = out.payload["messages"][-1]["content"]
    order = [d for d in (f'id="d{i}"' for i in range(6))]
    pos = sorted(order, key=content.find)
    # most relevant (shortest, d0) first; second-best (d1) last
    assert pos[0] == 'id="d0"' and pos[-1] == 'id="d1"'


def test_required_blocks_survive_and_optional_dropped_by_priority():
    blocks = [
        system_block("S " * 50),
        ContextBlock(name="low", kind=BlockKind.MEMORY, priority=10, min_tokens=60,
                     items=[ContextItem(id="m", text="memory " * 50)]),
        ContextBlock(name="high", kind=BlockKind.RETRIEVAL, priority=90, min_tokens=60,
                     items=[ContextItem(id="d", text="doc " * 50)]),
    ]
    out = ContextAssembler(window=170, reserve_output=20).assemble(blocks, query="question?")
    rep = out.report
    assert rep.block("system").status == "full"
    assert rep.block("high").items_kept == 1
    assert rep.block("low").status == "dropped"


def test_redistribution_hands_unused_budget_to_hungry_blocks():
    # Tools block is all-or-nothing and gets a share it cannot use; that slack
    # must flow to history rather than being wasted.
    big_tool = {"name": "t", "description": "x " * 700, "input_schema": {"type": "object"}}
    blocks = [
        tools_block([big_tool], weight=5.0, strategy=Strategy.DROP),
        history_block([{"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i} " * 10}
                       for i in range(40)], strategy=Strategy.DROP_OLDEST),
    ]
    out = ContextAssembler(window=700, reserve_output=100).assemble(blocks, query="hi")
    rep = out.report
    assert rep.block("tools").status == "dropped"
    assert rep.redistribution_rounds >= 1
    # slack reached history: only message granularity (~2 messages incl. user-first rule) is left over
    assert rep.block("history").used > 400 and rep.headroom < 60


def test_emergency_truncates_required_blocks():
    blocks = [system_block("instruction " * 400)]
    out = ContextAssembler(window=300, reserve_output=50).assemble(blocks, query="question " * 200)
    rep = out.report
    assert rep.payload_tokens <= rep.input_limit
    assert any("required" in w for w in rep.warnings)
    assert out.payload["messages"][-1]["content"]


def test_tiny_window_raises():
    with pytest.raises(ContextOverflowError):
        ContextAssembler(window=20, reserve_output=10).assemble([], query="hi")
    with pytest.raises(ValueError):
        ContextAssembler(window=0)


def test_duplicate_names_rejected():
    with pytest.raises(ValueError):
        ContextAssembler().assemble([system_block("a"), system_block("b")])


def test_llm_summarizer_plugged_in():
    llm = MockLLM().on(r"Summarize", "The customer asked repeatedly about late orders and refunds.")
    asm = ContextAssembler(window=900, reserve_output=200, summarizer=LLMSummarizer(llm))
    out = asm.assemble(scenario(), query=QUERY)
    first = out.payload["messages"][0]["content"]
    assert "late orders and refunds" in first and llm.calls
    assert out.report.payload_tokens <= out.report.input_limit


def test_deterministic():
    a = ContextAssembler(window=1500, reserve_output=300).assemble(scenario(), query=QUERY)
    b = ContextAssembler(window=1500, reserve_output=300).assemble(scenario(), query=QUERY)
    assert a.payload == b.payload


def test_safety_margin_and_custom_counter():
    def strict_counter(text: str) -> int:  # pessimistic tokenizer: 1 token per 3 chars
        return (len(text) + 2) // 3

    asm = ContextAssembler(window=1500, reserve_output=300, counter=strict_counter, safety_margin=100)
    out = asm.assemble(scenario(), query=QUERY)
    assert count_payload_tokens(out.payload, strict_counter) <= 1500 - 300 - 100


# ------------------------------------------------------------------ property test
WORDS = ("alpha beta gamma delta refund policy order shipping kafka latency extraordinarily "
         "internationalization a I x 42 3.14 $5 — ü 😀 , . ; ( ) { } <tag> \"quote\"").split()


def _rand_text(rng: random.Random, max_words: int) -> str:
    n = rng.randint(0, max_words)
    sep = rng.choice([" ", "  ", "\n", ""])
    return sep.join(rng.choice(WORDS) for _ in range(n))


def _rand_blocks(rng: random.Random) -> list[ContextBlock]:
    blocks = []
    for bi in range(rng.randint(0, 8)):
        kind = rng.choice(list(BlockKind))
        strategy = rng.choice(list(Strategy))
        items = []
        for ii in range(rng.randint(0, 12)):
            kw = dict(id=f"b{bi}i{ii}", text=_rand_text(rng, 150), pinned=rng.random() < 0.1,
                      score=rng.choice([None, rng.random()]), importance=rng.random(),
                      timestamp=rng.choice([None, float(ii)]))
            if kind == BlockKind.HISTORY:
                kw["role"] = rng.choice(["user", "assistant"])
            if kind == BlockKind.TOOL_DEFS:
                kw["tool"] = {"name": f"tool{bi}_{ii}", "description": _rand_text(rng, 40),
                              "input_schema": {"type": "object"}}
            if rng.random() < 0.2:
                kw["metadata"] = {"source": _rand_text(rng, 3)}
            items.append(ContextItem(**kw))
        lo = rng.choice([None, 0, rng.randint(0, 300)])
        hi = rng.choice([None, rng.randint(lo or 0, 2000)])
        blocks.append(ContextBlock(
            name=f"b{bi}", kind=kind, items=items, priority=rng.randint(0, 100), required=rng.random() < 0.3,
            min_tokens=lo, max_tokens=hi, weight=rng.choice([0.0, 0.5, 1.0, 3.0]), strategy=strategy,
            placement=rng.choice(list(Placement)), top_k=rng.choice([None, 1, 3]),
            mmr_lambda=rng.choice([None, 0.5]), allow_partial=rng.random() < 0.7,
            summary_ratio=rng.random(),
        ))
    return blocks


@pytest.mark.parametrize("seed", range(300))
def test_never_exceeds_window(seed):
    rng = random.Random(seed)
    window = rng.choice([30, 64, 128, 256, 512, 1024, 4096, 20_000])
    reserve = rng.randint(0, window // 2)
    asm = ContextAssembler(window=window, reserve_output=reserve, safety_margin=rng.choice([0, 0, 5]),
                           cache_static_prefix=rng.random() < 0.5)
    blocks = _rand_blocks(rng)
    query = _rand_text(rng, 80) if rng.random() < 0.8 else None
    try:
        out = asm.assemble(blocks, query=query)
    except ContextOverflowError:
        assert asm.input_limit - 7 < 8  # only legitimate when the window cannot hold framing
        return
    measured = count_payload_tokens(out.payload, count_tokens)
    assert measured <= asm.input_limit
    assert measured == out.report.payload_tokens
    assert out.report.repair_rounds == 0  # the cost model is an exact upper bound
    msgs = out.payload["messages"]
    assert msgs and msgs[0]["role"] == "user" and msgs[-1]["role"] == "user"
    assert all(a["role"] != b["role"] for a, b in zip(msgs, msgs[1:]))
    tools = out.payload.get("tools", [])
    assert len({t["name"] for t in tools}) == len(tools)


def test_benchmark_components_are_deterministic_and_serializable():
    from context_assembler import bench

    small = bench.retention_benchmark(n_cases=4)
    assert json.dumps(small) and small == bench.retention_benchmark(n_cases=4)
    rates = small["retention_rates"]
    assert rates["assembler"]["all_four"] == 1.0
    assert rates["naive_keep_head"]["query"] == 0.0
    ob = bench.overflow_benchmark(n_cases=30)
    assert ob["overflows"] == 0 and ob["repair_rounds"] == 0
