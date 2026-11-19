"""Allocator, formatting/fitting, placement, summarizers and compressors."""
from __future__ import annotations

import pytest

from aikit import HashingEmbedder, MockLLM, count_tokens
from context_assembler import (
    BlockDemand, BlockKind, BudgetAllocator, CompressionContext, ContextBlock, ContextItem, ExtractiveSummarizer,
    Formatter, LLMSummarizer, Placement, Strategy, compress, edge_order, fit_middle, fit_prefix, history_block,
    proportional_split, retrieval_block, tool_results_block, tools_block, water_fill,
)
from context_assembler.formatting import TRUNCATION_MARK


# ------------------------------------------------------------------ allocator
def test_proportional_split_is_exact():
    out = proportional_split(10, {"a": 1, "b": 1, "c": 1})
    assert sum(out.values()) == 10 and max(out.values()) - min(out.values()) <= 1
    assert proportional_split(0, {"a": 1}) == {"a": 0}
    assert sum(proportional_split(7, {"a": 0, "b": 0}).values()) == 7


def test_water_fill_redistributes_saturated_share():
    out = water_fill(100, {"a": 10, "b": 1000}, {"a": 1, "b": 1})
    assert out == {"a": 10, "b": 90}
    assert water_fill(50, {"a": 5, "b": 5}, {"a": 1, "b": 1}) == {"a": 5, "b": 5}


def test_allocator_required_then_priority_then_weight():
    demands = [
        BlockDemand("sys", 100, True, floor=50, cap=50),
        BlockDemand("q", 100, True, floor=10, cap=10),
        BlockDemand("docs", 70, False, floor=30, cap=500, weight=3),
        BlockDemand("hist", 60, False, floor=0, cap=500, weight=1),
        BlockDemand("mem", 10, False, floor=200, cap=200),
    ]
    a = BudgetAllocator().allocate(demands, 260)
    assert a.tokens["sys"] == 50 and a.tokens["q"] == 10
    assert a.tokens["mem"] == 0 and a.dropped == ["mem"]
    # surplus 260-60-30 = 170 split 3:1 between docs and hist
    assert a.tokens["docs"] == 30 + 128 or a.tokens["docs"] == 30 + 127
    assert a.total == 260


def test_allocator_respects_caps_and_never_overspends():
    demands = [BlockDemand("a", 1, False, 0, 5), BlockDemand("b", 1, False, 0, 7)]
    a = BudgetAllocator().allocate(demands, 1000)
    assert a.tokens == {"a": 5, "b": 7}


def test_allocator_emergency_scales_required():
    demands = [BlockDemand("sys", 100, True, 300, 300), BlockDemand("q", 100, True, 100, 100)]
    a = BudgetAllocator().allocate(demands, 200)
    assert a.total == 200 and a.warnings
    assert a.tokens["sys"] > a.tokens["q"] > 0


def test_allocator_rejects_duplicates():
    with pytest.raises(ValueError):
        BudgetAllocator().allocate([BlockDemand("a", 1, False, 0, 1)] * 2, 10)


# ------------------------------------------------------------------ fitting / formatting
LONG = " ".join(f"word{i}" for i in range(400))


@pytest.mark.parametrize("budget", [1, 5, 17, 60, 200])
def test_fit_functions_respect_budget(budget):
    for fn in (fit_prefix, fit_middle):
        out = fn(LONG, budget, count_tokens)
        assert count_tokens(out) <= budget


def test_fit_middle_keeps_head_and_tail():
    out = fit_middle(LONG, 80, count_tokens)
    assert out.startswith("word0") and out.endswith("word399") and TRUNCATION_MARK in out


def test_formatter_costs_are_additive():
    f = Formatter()
    items = [ContextItem(id=f"d{i}", text=f"Document {i} text about pricing.", metadata={"source": "kb"}) for i in range(5)]
    assert f.block_cost(BlockKind.RETRIEVAL, items) == count_tokens(f.render_block(BlockKind.RETRIEVAL, items))
    assert f.block_cost(BlockKind.RETRIEVAL, []) == 0


# ------------------------------------------------------------------ placement
def test_edge_order_puts_best_at_edges():
    assert edge_order([1, 2, 3, 4, 5]) == [1, 3, 5, 4, 2]
    assert edge_order([]) == [] and edge_order([1]) == [1]


# ------------------------------------------------------------------ summarizers
TEXT = ("The migration to Kafka finished on March 3. The team spent $40k on new brokers. "
        "Weather was nice that week. Latency dropped by 35% after the change. Someone brought donuts.")


@pytest.mark.parametrize("budget", [3, 10, 25, 1000])
def test_extractive_summary_within_budget(budget):
    out = ExtractiveSummarizer().summarize([TEXT], budget)
    assert count_tokens(out) <= budget
    if budget >= 1000:
        assert out.count(".") == 5


def test_extractive_summary_is_query_aware_and_keeps_labels():
    s = ExtractiveSummarizer().summarize(["User: " + TEXT], 14, query="how much did brokers cost")
    assert "$40k" in s and s.startswith("User:")


def test_llm_summarizer_enforces_budget_and_falls_back():
    llm = MockLLM().on(r"Summarize", "word " * 500)
    s = LLMSummarizer(llm).summarize([TEXT], 20)
    assert count_tokens(s) <= 20 and llm.calls

    class Boom:
        model = "boom"

        def complete(self, *a, **k):
            raise RuntimeError("down")

    summ = LLMSummarizer(Boom())
    out = summ.summarize([TEXT], 20)
    assert summ.failures == 1 and out and count_tokens(out) <= 20


# ------------------------------------------------------------------ compressors
@pytest.fixture
def ctx():
    emb = HashingEmbedder()
    return CompressionContext(Formatter(), emb, ExtractiveSummarizer(embedder=emb), query="refund policy for returns")


def _hist(n: int, start_role: str = "user"):
    roles = ["user", "assistant"] if start_role == "user" else ["assistant", "user"]
    return history_block([{"role": roles[i % 2], "content": f"Turn {i}: details about item {i} and more words."}
                          for i in range(n)])


def test_drop_oldest_keeps_newest_and_user_first(ctx):
    b = _hist(20).model_copy(update={"strategy": Strategy.DROP_OLDEST})
    r = compress(b, 80, ctx)
    f = ctx.formatter
    assert f.block_cost(b.kind, r.items) <= 80
    assert r.items[-1].id == "msg19" and r.items[0].role == "user"
    ids = [int(i.id[3:]) for i in r.items]
    assert ids == list(range(ids[0], 20))
    assert "msg0" in r.dropped


def test_drop_oldest_pins(ctx):
    b = _hist(20).model_copy(update={"strategy": Strategy.DROP_OLDEST})
    b.items[0].pinned = True
    r = compress(b, 80, ctx)
    assert r.items[0].id == "msg0"


def test_summarize_history_inserts_summary(ctx):
    b = _hist(30)
    r = compress(b, 150, ctx)
    assert ctx.formatter.block_cost(b.kind, r.items) <= 150
    assert r.items[0].id == "history:summary" and r.items[0].role == "user"
    assert r.compressed and r.items[-1].id == "msg29"
    assert not set(r.compressed) & set(r.dropped)


def test_history_fits_untouched(ctx):
    b = _hist(4)
    r = compress(b, 10_000, ctx)
    assert [i.id for i in r.items] == ["msg0", "msg1", "msg2", "msg3"] and not r.lossy


def test_history_leading_assistant_dropped(ctx):
    b = _hist(5, start_role="assistant")
    r = compress(b, 10_000, ctx)
    assert r.items[0].role == "user" and r.dropped == ["msg0"]


def _docs():
    docs = [{"id": f"d{i}", "text": "Kafka consumer lag tuning and broker partitions. " * 6} for i in range(10)]
    docs[6] = {"id": "needle", "text": "Our refund policy allows returns within 30 days with a receipt."}
    return docs


def test_top_k_keeps_relevant_needle(ctx):
    b = retrieval_block(_docs(), top_k=3)
    r = compress(b, 200, ctx)
    ids = [i.id for i in r.items]
    assert "needle" in ids and len(ids) <= 3
    assert ctx.formatter.block_cost(b.kind, r.items) <= 200
    assert r.items[0].id == "needle"  # best item first under EDGES placement


def test_mmr_diversifies(ctx):
    docs = [{"id": f"dup{i}", "text": "refund policy returns thirty days receipt"} for i in range(4)]
    docs.append({"id": "other", "text": "refund for digital goods is store credit only"})
    plain = compress(retrieval_block(docs, top_k=2, mmr_lambda=None), 10_000, ctx)
    diverse = compress(retrieval_block(docs, top_k=2, mmr_lambda=0.5), 10_000, ctx)
    assert "other" not in [i.id for i in plain.items]
    assert "other" in [i.id for i in diverse.items]


def test_top_k_partial_truncation(ctx):
    b = retrieval_block([{"id": "big", "text": "refund policy " + "detail " * 400}])
    r = compress(b, 100, ctx)
    assert r.compressed == ["big"] and ctx.formatter.block_cost(b.kind, r.items) <= 100


def test_tools_selected_by_relevance_never_truncated(ctx):
    tools = [
        {"name": "get_weather", "description": "Get the weather forecast for a city",
         "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}},
        {"name": "issue_refund", "description": "Issue a refund for an order return",
         "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}}},
        {"name": "send_email", "description": "Send an email message",
         "input_schema": {"type": "object", "properties": {"to": {"type": "string"}}}},
    ]
    b = tools_block(tools, pinned=["send_email"])
    one = ctx.formatter.item_cost(b.kind, b.items[0])
    r = compress(b, 2 * one + 2, ctx)
    names = {i.id for i in r.items}
    assert names == {"issue_refund", "send_email"}
    assert all(i.tool == next(t for t in tools if t["name"] == i.id) for i in r.items)


def test_truncate_middle_tool_results(ctx):
    b = tool_results_block([{"name": "sql", "content": "row " * 500}, {"name": "grep", "content": "short output"}])
    r = compress(b, 120, ctx)
    assert ctx.formatter.block_cost(b.kind, r.items) <= 120
    assert [i.text for i in r.items if i.id != "sql"] == ["short output"]  # small item untouched
    assert "sql" in r.compressed


def test_truncate_and_drop_strategies(ctx):
    items = [ContextItem(id=f"s{i}", text="alpha beta gamma " * 10) for i in range(3)]
    b = ContextBlock(name="sys", kind=BlockKind.SYSTEM, items=items, strategy=Strategy.TRUNCATE)
    one = ctx.formatter.item_cost(b.kind, items[0])
    budget = one + one // 2
    r = compress(b, budget, ctx)
    assert [i.id for i in r.items] == ["s0", "s1"] and r.compressed == ["s1"] and r.dropped == ["s2"]
    d = compress(b.model_copy(update={"strategy": Strategy.DROP}), budget, ctx)
    assert d.items == [] and len(d.dropped) == 3
    n = compress(b.model_copy(update={"strategy": Strategy.NONE}), budget, ctx)
    assert [i.id for i in n.items] == ["s0"]


def test_summarize_non_history(ctx):
    b = ContextBlock(name="notes", kind=BlockKind.TOOL_RESULT, strategy=Strategy.SUMMARIZE,
                     items=[ContextItem(id="n", text=TEXT * 5)])
    r = compress(b, 40, ctx)
    assert len(r.items) == 1 and ctx.formatter.block_cost(b.kind, r.items) <= 40 and r.compressed == ["n"]


def test_block_validation():
    with pytest.raises(ValueError):
        ContextBlock(name="h", kind=BlockKind.HISTORY, items=[ContextItem(id="x", text="hi")])
    with pytest.raises(ValueError):
        ContextBlock(name="t", kind=BlockKind.TOOL_DEFS, items=[ContextItem(id="x")])
    with pytest.raises(ValueError):
        ContextBlock(name="b", kind=BlockKind.MEMORY, min_tokens=10, max_tokens=5)
    assert ContextBlock(name="p", kind=BlockKind.RETRIEVAL, placement=Placement.EDGES).weight == 1.0
