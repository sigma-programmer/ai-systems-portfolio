"""The Context Assembler: budget -> compress -> redistribute -> place -> render -> verify."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from aikit import HashingEmbedder, count_tokens

from .allocator import BlockDemand, BudgetAllocator, water_fill
from .compressors import CompressionContext, CompressionResult, compress
from .formatting import BASE_OVERHEAD, MESSAGE_OVERHEAD, Counter, Formatter, fit_item
from .models import BlockKind, ContextBlock, ContextItem, ContextOverflowError, Strategy
from .render import count_payload_tokens, render_payload, to_prompt
from .report import AssemblyReport, BlockReport
from .summarize import ExtractiveSummarizer, Summarizer

#: Smallest input budget we are willing to assemble into.
MIN_BUDGET = 8


@dataclass
class AssembledContext:
    """The rendered payload plus its observability report."""

    payload: dict[str, Any]
    report: AssemblyReport

    def to_prompt(self) -> tuple[str, str]:
        """``(system, prompt)`` for the text-only :class:`aikit.LLM` protocol."""
        return to_prompt(self.payload)

    def request(self, model: str) -> dict[str, Any]:
        """Keyword arguments for ``anthropic.Anthropic().messages.create(**...)``."""
        return {"model": model, **self.payload}


class ContextAssembler:
    """Fit heterogeneous context sources into a fixed window, deterministically.

    Args:
        window: model context window in tokens.
        reserve_output: tokens reserved for the response (``max_tokens``).
        counter: token counter; must be monotone in text length. All costs,
            compressions and the final verification use this same function.
        summarizer: used by ``Strategy.SUMMARIZE`` (extractive by default; pass
            :class:`~context_assembler.summarize.LLMSummarizer` for abstractive).
        embedder: relevance model for TOP_K reranking and EDGES placement.
        safety_margin: extra tokens kept free, to absorb tokenizer mismatch
            against the provider's real tokenizer.
        cache_static_prefix: mark the system prompt / tools for prompt caching.
        redistribution_rounds: passes that hand unused allocation to blocks that
            were compressed.
    """

    def __init__(
        self,
        window: int = 200_000,
        reserve_output: int = 4096,
        counter: Counter = count_tokens,
        summarizer: Summarizer | None = None,
        embedder: HashingEmbedder | None = None,
        safety_margin: int = 0,
        cache_static_prefix: bool = True,
        redistribution_rounds: int = 2,
    ):
        if window <= 0 or reserve_output < 0 or safety_margin < 0:
            raise ValueError("window must be > 0, reserve_output and safety_margin >= 0")
        self.window = window
        self.reserve_output = reserve_output
        self.counter = counter
        self.formatter = Formatter(counter)
        self.embedder = embedder or HashingEmbedder()
        self.summarizer = summarizer or ExtractiveSummarizer(counter, self.embedder)
        self.safety_margin = safety_margin
        self.cache_static_prefix = cache_static_prefix
        self.redistribution_rounds = redistribution_rounds
        self.allocator = BudgetAllocator()

    # ------------------------------------------------------------------ public
    @property
    def input_limit(self) -> int:
        return self.window - self.reserve_output - self.safety_margin

    def assemble(self, blocks: list[ContextBlock], query: str | None = None) -> AssembledContext:
        """Assemble ``blocks`` (plus an optional query string) into a payload that fits.

        Raises:
            ContextOverflowError: if the window minus reserve cannot hold even
                the request framing.
            ValueError: on duplicate block names.
        """
        t0 = time.perf_counter()
        blocks = list(blocks)
        if query is not None:
            blocks.append(ContextBlock(name="query", kind=BlockKind.QUERY, required=True, priority=100,
                                       strategy=Strategy.TRUNCATE_MIDDLE,
                                       items=[ContextItem(id="query", text=query)]))
        names = [b.name for b in blocks]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate block names: {names}")
        limit = self.input_limit
        budget = limit - BASE_OVERHEAD - MESSAGE_OVERHEAD  # framing of the final user turn
        if budget < MIN_BUDGET:
            raise ContextOverflowError(
                f"window {self.window} - reserve {self.reserve_output} - margin {self.safety_margin} "
                f"leaves {budget} tokens; need at least {MIN_BUDGET}"
            )
        query_text = " ".join(it.text for b in blocks if b.kind == BlockKind.QUERY for it in b.items)
        ctx = CompressionContext(self.formatter, self.embedder, self.summarizer, query=query_text)

        demands = {b.name: self._demand(b) for b in blocks}
        payload, results, alloc_tokens, warnings, rounds = self._run(blocks, demands, budget, ctx)
        measured = count_payload_tokens(payload, self.counter)
        repairs = 0
        # Verify-and-repair. The cost model is an exact upper bound, so this loop
        # is defence in depth (e.g. a non-additive custom counter).
        while measured > limit and repairs < 3:
            repairs += 1
            budget -= measured - limit
            payload, results, alloc_tokens, warnings, rounds = self._run(blocks, demands, max(0, budget), ctx)
            measured = count_payload_tokens(payload, self.counter)
        while measured > limit:  # last resort: shed lowest-priority content item by item
            repairs += 1
            if not self._shed_one(blocks, results):
                raise ContextOverflowError("cannot fit even an empty payload")
            payload = self._render(blocks, results)
            measured = count_payload_tokens(payload, self.counter)
        if repairs:
            warnings.append(f"cost model under-estimated; repaired in {repairs} round(s)")

        report = self._report(blocks, demands, alloc_tokens, results, payload, measured, warnings, rounds, repairs)
        report.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        return AssembledContext(payload=payload, report=report)

    # ------------------------------------------------------------------ phases
    def _demand(self, b: ContextBlock) -> BlockDemand:
        f = self.formatter
        demand = f.block_cost(b.kind, b.items)
        cap = demand if b.max_tokens is None else min(demand, b.max_tokens)
        pinned = [it for it in b.items if it.pinned]
        pinned_cost = f.block_cost(b.kind, pinned)
        if b.min_tokens is None:
            floor = cap if b.required else pinned_cost
        else:
            floor = max(b.min_tokens, pinned_cost)
        return BlockDemand(b.name, b.priority, b.required, min(floor, cap), cap, b.weight)

    def _run(self, blocks, demands, budget, ctx):
        alloc = self.allocator.allocate(list(demands.values()), budget)
        tokens = dict(alloc.tokens)
        warnings = list(alloc.warnings)
        by_name = {b.name: b for b in blocks}
        results: dict[str, CompressionResult] = {
            b.name: compress(b, tokens[b.name], ctx) for b in blocks
        }
        rounds = 0
        stalled: set[str] = set()  # blocks that could not convert extra budget into content
        attempts = 0
        while rounds < self.redistribution_rounds and attempts < self.redistribution_rounds + len(blocks):
            attempts += 1
            used = {n: self.formatter.block_cost(by_name[n].kind, r.items) for n, r in results.items()}
            slack = budget - sum(used.values())
            # Blocks that lost content and could use more; dropped optional blocks
            # are eligible again if slack now covers their floor.
            hungry = {
                n: demands[n].cap - used[n] for n, r in results.items()
                if n not in stalled and r.lossy and demands[n].cap > used[n]
                and (tokens[n] > 0 or demands[n].floor <= slack)
            }
            if slack <= 0 or not hungry:
                break
            extra = water_fill(slack, hungry, {n: max(by_name[n].weight, 1e-6) for n in hungry})
            changed = False
            for n in sorted(hungry):
                if extra[n] <= 0:
                    continue
                new_budget = used[n] + extra[n]
                trial = compress(by_name[n], new_budget, ctx)
                gained = self.formatter.block_cost(by_name[n].kind, trial.items) > used[n]
                if gained or len(trial.items) > len(results[n].items):
                    results[n] = trial
                    tokens[n] = new_budget
                    changed = True
                else:
                    stalled.add(n)  # its share is released to the others next pass
            if changed:
                rounds += 1
        return self._render(blocks, results), results, tokens, warnings, rounds

    def _render(self, blocks: list[ContextBlock], results: dict[str, CompressionResult]) -> dict[str, Any]:
        f = self.formatter
        system, memory, tools, history, final_docs, final_tools, final_query = [], [], [], [], [], [], []
        for b in blocks:
            items = results[b.name].items
            if not items:
                continue
            if b.kind == BlockKind.SYSTEM:
                system.append(f.render_block(b.kind, items))
            elif b.kind == BlockKind.MEMORY:
                memory.append(f.render_block(b.kind, items))
            elif b.kind == BlockKind.TOOL_DEFS:
                tools.extend(it.tool for it in items)
            elif b.kind == BlockKind.HISTORY:
                history.extend((it.role, it.text) for it in items)
            elif b.kind == BlockKind.RETRIEVAL:
                final_docs.append(f.render_block(b.kind, items))
            elif b.kind == BlockKind.TOOL_RESULT:
                final_tools.append(f.render_block(b.kind, items))
            else:
                final_query.append(f.render_block(b.kind, items))
        return render_payload(system, memory, tools, history, final_docs + final_tools + final_query,
                              self.reserve_output, self.cache_static_prefix)

    def _shed_one(self, blocks: list[ContextBlock], results: dict[str, CompressionResult]) -> bool:
        """Remove (or halve) the least important remaining item. False if nothing is left."""
        order = sorted(blocks, key=lambda b: (b.required, b.priority, b.name))
        for b in order:
            r = results[b.name]
            if not r.items:
                continue
            victim = r.items[-1] if b.kind != BlockKind.HISTORY else r.items[0]
            cost = self.formatter.item_cost(b.kind, victim)
            cut = fit_item(self.formatter, b.kind, victim, cost // 2, mode="middle") if b.required else None
            idx = r.items.index(victim)
            if cut is not None and cut.text != victim.text:
                r.items[idx] = cut
                r.compressed.append(victim.id)
            else:
                r.items.pop(idx)
                r.dropped.append(victim.id)
            r.actions.append(f"emergency shed of {victim.id!r}")
            return True
        return False

    def _report(self, blocks, demands, tokens, results, payload, measured, warnings, rounds, repairs) -> AssemblyReport:
        f = self.formatter
        reports, by_kind = [], {}
        estimated = BASE_OVERHEAD + MESSAGE_OVERHEAD
        for b in blocks:
            r = results[b.name]
            used = f.block_cost(b.kind, r.items)
            estimated += used
            by_kind[b.kind.value] = by_kind.get(b.kind.value, 0) + used
            d = demands[b.name]
            reports.append(BlockReport(
                name=b.name, kind=b.kind.value, priority=b.priority, required=b.required,
                strategy=b.strategy.value, demand=f.block_cost(b.kind, b.items), floor=d.floor,
                allocated=tokens[b.name], used=used, items_in=len(b.items), items_kept=len(r.items),
                dropped_ids=list(dict.fromkeys(r.dropped)), compressed_ids=list(dict.fromkeys(r.compressed)),
                actions=list(r.actions),
            ))
            if b.required and (r.dropped or r.compressed):
                warnings.append(f"required block {b.name!r} was compressed")
        return AssemblyReport(
            window=self.window, reserve_output=self.reserve_output, safety_margin=self.safety_margin,
            budget=self.input_limit - BASE_OVERHEAD - MESSAGE_OVERHEAD,
            estimated_tokens=estimated, payload_tokens=measured, by_kind=by_kind, blocks=reports,
            warnings=list(dict.fromkeys(warnings)), redistribution_rounds=rounds, repair_rounds=repairs,
        )
