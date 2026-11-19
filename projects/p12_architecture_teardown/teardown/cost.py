"""Cost-per-task analysis.

A *task* (e.g. "answer one support question with agentic RAG") is modelled as
a list of LLM calls with token counts. We price each call against a model
price table (USD per million tokens, first-party Anthropic list prices) and
compare deployment strategies: always-Opus, complexity-routed, and
routed + semantic cache + prompt caching.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelPrice:
    input: float          # $ / MTok
    output: float         # $ / MTok
    cache_read: float     # $ / MTok


PRICING: dict[str, ModelPrice] = {
    "claude-opus-5-5": ModelPrice(4.00, 20.00, 0.20),
    "claude-sonnet-5-5": ModelPrice(2.00, 10.00, 0.10),
    "claude-haiku-5-5": ModelPrice(0.10, 0.50, 0.01),
}

TIER_MODEL = {"high": "claude-opus-5-5", "medium": "claude-sonnet-5-5", "low": "claude-haiku-5-5"}


@dataclass
class LLMCall:
    purpose: str
    input_tokens: int
    output_tokens: int
    tier: str = "high"              # complexity tier the router would assign
    cacheable_prefix: int = 0       # tokens of stable prefix eligible for prompt caching
    model: str | None = None        # explicit model overrides tier routing


@dataclass
class TaskProfile:
    name: str
    calls: list[LLMCall] = field(default_factory=list)
    semantic_cache_hit_rate: float = 0.0  # fraction of whole tasks served from cache


def cost_of_call(call: LLMCall, model: str, prompt_cache: bool = False) -> float:
    if model not in PRICING:
        raise KeyError(f"unknown model {model!r}")
    p = PRICING[model]
    cached = min(call.cacheable_prefix, call.input_tokens) if prompt_cache else 0
    fresh = call.input_tokens - cached
    return (fresh * p.input + cached * p.cache_read + call.output_tokens * p.output) / 1e6


def task_cost(task: TaskProfile, strategy: str) -> float:
    """Expected USD cost of one task under a strategy.

    strategies: "always_opus", "routed", "routed_cached"
    """
    if strategy not in {"always_opus", "routed", "routed_cached"}:
        raise ValueError(strategy)
    total = 0.0
    for c in task.calls:
        model = c.model or ("claude-opus-5-5" if strategy == "always_opus" else TIER_MODEL[c.tier])
        total += cost_of_call(c, model, prompt_cache=strategy == "routed_cached")
    if strategy == "routed_cached":
        total *= 1.0 - task.semantic_cache_hit_rate
    return total


def compare_strategies(tasks: list[TaskProfile]) -> list[dict]:
    rows = []
    for t in tasks:
        base = task_cost(t, "always_opus")
        r = task_cost(t, "routed")
        rc = task_cost(t, "routed_cached")
        rows.append({
            "task": t.name,
            "llm_calls": len(t.calls),
            "tokens": sum(c.input_tokens + c.output_tokens for c in t.calls),
            "always_opus_usd": round(base, 6),
            "routed_usd": round(r, 6),
            "routed_cached_usd": round(rc, 6),
            "savings_pct": round(100 * (1 - rc / base), 1) if base else 0.0,
        })
    return rows


def reference_tasks() -> list[TaskProfile]:
    """Token profiles of representative tasks for each subsystem.

    Token counts reflect the prompt shapes the projects actually build
    (context assembler output sizes, retrieval top-k * chunk size, etc.).
    """
    return [
        TaskProfile("Agentic RAG answer (p04)", [
            LLMCall("query rewrite", 400, 80, "low", 300),
            LLMCall("relevance grading x5", 3000, 100, "low", 2500),
            LLMCall("grounded generation", 6000, 500, "medium", 1500),
            LLMCall("citation verification", 2500, 150, "low", 800),
        ], semantic_cache_hit_rate=0.25),
        TaskProfile("GraphRAG ingest per doc (p03)", [
            LLMCall("entity+relation extraction", 1800, 600, "medium", 1000),
            LLMCall("community summary (amortized)", 900, 200, "low", 400),
        ]),
        TaskProfile("Swarm codegen task (p08)", [
            LLMCall("supervisor plan", 1500, 400, "high", 1000),
            LLMCall("research", 4000, 800, "medium", 1200),
            LLMCall("code x2 revisions", 9000, 2400, "high", 3000),
            LLMCall("review x3 voters", 9000, 900, "medium", 3000),
        ]),
        TaskProfile("Memory consolidation per session (p07)", [
            LLMCall("episode compression", 3000, 300, "low", 500),
            LLMCall("fact extraction", 1200, 200, "low", 500),
        ]),
        TaskProfile("Security proxy LLM classifier (p10)", [
            LLMCall("injection classification", 900, 30, "low", 700),
        ]),
        TaskProfile("Trajectory judge per PR (p09)", [
            LLMCall("step-level judge x40", 48000, 4000, "medium", 36000),
        ]),
    ]
