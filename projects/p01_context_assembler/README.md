# P01 — Context Assembler Engine

> You master the context window, not fight it.

A deterministic engine that takes every candidate source of context and fits
it into a fixed model window with a **hard guarantee**: the assembled payload
never exceeds `window - reserve_output`. The sources are system prompt,
conversation history, long-term memories, retrieved documents, tool
definitions, tool results and the user query. The engine decides what goes
in, what gets compressed and what gets dropped, and in what order. It
explains every decision in an assembly report and renders the result as an
Anthropic Messages-API payload.

## Problem

Agents accumulate more context than any window holds. Naive strategies fail
in predictable ways:

- **Keep the head:** the user's question falls off the end.
- **Keep the tail:** the system prompt and the facts from early turns are lost.
- **Hard-code per-source caps:** budget is wasted when one source is small
  and starved when another is large.
- **Pass everything:** the request either overflows the window or is
  needlessly expensive.

Every naive variant also eventually ships an over-long request, because the
token estimate and the rendered payload drift apart.

## Architecture

```
  ContextBlocks (typed sources, each with priority / required / min / max / weight / strategy)
  ┌────────┬─────────┬─────────┬──────────┬───────────┬──────────────┬───────┐
  │ system │ history │ memory  │ retrieval│ tool_defs │ tool_results │ query │
  └───┬────┴────┬────┴────┬────┴────┬─────┴─────┬─────┴──────┬───────┴───┬───┘
      └─────────┴─────────┴────┬────┴───────────┴────────────┴───────────┘
                               ▼
   Formatter: exact, additive cost of every rendered fragment (same counter everywhere)
                               ▼
   BudgetAllocator   budget = window − reserve_output − safety_margin − framing
     1. required floors (priority order; emergency proportional scaling + warning)
     2. optional floors by priority (block dropped if its floor does not fit)
     3. surplus water-filled by weight up to each block's cap
                               ▼
   Compressors (per block, output cost ≤ allocation, self-verified)
     truncate · truncate_middle · drop_oldest · summarize (extractive | LLM)
     · top_k (relevance rerank + MMR, partial last item) · drop · none
                               ▼
   Redistribution: unused allocation → blocks that lost content (re-compress)
                               ▼
   Placement: system/tools first (cacheable prefix) · memories in system ·
     history chronological · docs EDGE-ordered (best first, 2nd-best last) ·
     tool results · query LAST
                               ▼
   Renderer → {system:[{text, cache_control}], tools:[...], messages:[...], max_tokens}
                               ▼
   Verify: count_payload_tokens ≤ limit  (else repair loop → emergency shed)
                               ▼
   AssembledContext(payload, AssemblyReport: per-block demand/floor/alloc/used,
                    kept/dropped/compressed ids, actions, tokens by kind, warnings)
```

## Key design decisions and tradeoffs

- **Correct by construction, verified anyway.** The allocator, compressors,
  renderer and verifier all price text with the *same* counter, through one
  `Formatter`. Fragments are joined with whitespace and the counter splits on
  whitespace, so section cost equals the sum of fragment costs. Message
  framing is charged at an upper bound (4 tokens per message, even though
  adjacent same-role turns get merged).
  - **Testing:** the property test over 300 randomized configurations (and
    3,000 more run ad hoc) checks `measured ≤ limit`, `measured == estimate`
    and `repair_rounds == 0`. The randomization covers all block kinds and
    strategies, unicode/emoji/punctuation text, tiny windows and pinned items.
  - **Fallback:** a verify-and-repair loop, then emergency shedding,
    guarantees the bound even with a non-additive custom tokenizer (also
    tested).
- **Three-phase allocation with integer water-filling.** Required content
  (system prompt, query) is protected first. Optional blocks compete by
  priority for their floors, then share the surplus by weight. A saturated
  block releases its share instead of wasting it. All arithmetic is integer
  with largest-remainder rounding, so allocations never sum past the budget.
- **Redistribution after compression.** Allocation runs before compression,
  but compression is lumpy: a tool definition is all-or-nothing, `top_k`
  caps a block, and messages are discrete. A second pass hands the unused
  tokens to blocks that lost content. Blocks that can't turn extra budget
  into content are marked stalled, so their share goes to the others.
- **Strategies matched to each source's semantics:**
  - **History:** sliding window plus a summary of what fell out. The summary
    is extractive by default and query-aware, so a fact stated 70 turns ago
    survives. Pinned turns stay. Assistant-first openings are dropped to
    satisfy the API's user-first rule.
  - **Retrieval:** reranked by query similarity, upstream score, importance
    and recency, with MMR for diversity. The last item may be truncated to
    fill the remaining budget.
  - **Tools:** selected by relevance, never truncated, because a cut schema
    is invalid. Pinned tools always stay.
  - **Tool results:** max-min-fair middle truncation that keeps the head and
    tail of each result, which is where status lines and final values sit.
- **Placement against lost-in-the-middle.** Ranked items are edge-ordered:
  best first, second-best last, weakest in the middle. The query is the last
  thing in the final user turn. The system prompt and tools form a stable
  prefix marked `cache_control: ephemeral` for prompt caching. Memories
  change per request, so they go in a separate uncached system block.
- **LLM-pluggable, offline-first.** `LLMSummarizer` uses any `aikit.LLM`.
  Its output is hard-fitted to the allocation, so a verbose model can't
  break the budget, and it falls back to the extractive summarizer if the
  provider errors. The default `ExtractiveSummarizer` scores sentences by
  query relevance, centrality, position and numeric/proper-noun signal, with
  a redundancy filter. That makes it a real algorithm, not canned text.
- **Tokenizer caveat.** `aikit.count_tokens` is an approximation that
  deliberately over-estimates. Against a real provider tokenizer, set
  `safety_margin` or pass `counter=`; any monotone counter works,
  including a pessimistic one used in the tests.

## How to run

```bash
AIKIT_OFFLINE=1 .venv/bin/python projects/p01_context_assembler/demo.py
.venv/bin/python -m pytest projects/p01_context_assembler            # 346 tests
cd projects/p01_context_assembler && ../../.venv/bin/python -m context_assembler.bench
```

```python
from context_assembler import *
asm = ContextAssembler(window=200_000, reserve_output=4096)        # summarizer=LLMSummarizer(llm) optional
out = asm.assemble([
    system_block("You are ..."), history_block(messages), memory_block(memories),
    retrieval_block(docs, top_k=8), tools_block(tools, pinned=["search"]),
    tool_results_block(results), query_block(user_query),
])
print(out.report.render_text())
anthropic.Anthropic().messages.create(**out.request("claude-opus-5-5"))  # or out.to_prompt() for aikit.LLM
```

## Benchmark (measured, `context_assembler.bench.benchmark()`, 12.6 s total)

**Overflow property**: 250 randomized configurations with windows from 64 to
16k tokens. All block kinds and strategies are mixed in, and blocks are
randomly marked required.

| overflows | repair rounds | constrained cases | utilization when constrained (median / mean) |
|---|---|---|---|
| **0** | 0 | 176 | 96.8% / 78.0% |

The mean is pulled down by configurations whose blocks are capped by `top_k`,
`max_tokens` or all-or-nothing `DROP`. In those cases the budget is
deliberately left unused rather than filled with low-value content.

**Information retention vs. naive truncation.** The test set is 30
long-support-conversation cases: 80 turns, 60 documents, a needle document
in a random position, and a code stated once in an early turn. Full context
averages 13,977 tokens; the window is 4,096 with 1,024 reserved.

| system | needle doc | buried fact | query | system prompt | all four |
|---|---|---|---|---|---|
| **assembler (summarize history)** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| assembler, drop-oldest history (ablation) | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 |
| naive keep-head truncation | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 |
| naive keep-tail truncation | 0.27 | 0.00 | 1.00 | 0.00 | 0.00 |

The ablation isolates the value of query-aware history summarization: a
pure sliding window loses the early fact every time.

**Latency.** The input is 300 turns, 200 documents, 8 selected tools and an
4k-token tool result, about 59k tokens of demand, measured warm:

| window | p50 | p95 | payload tokens | utilization |
|---|---|---|---|---|
| 8,192 | 43.7 ms | 45.1 ms | 5,992 | 97.5% |
| 32,768 | 9.8 ms | 10.2 ms | 19,870 | 64.7% (top-k caps reached; everything else fits) |
| 200,000 | 6.2 ms | 8.1 ms | 19,870 | 10.0% |

The tight-window case is slower because it truncates the 8k-token tool
result by binary search (the cost is the repeated counter calls) and
summarizes about 280 turns.

**Cost** at `claude-opus-5-5`'s $4 per million input tokens: $12.03 per 1k
requests, vs $55.91 to send the full context (**78% fewer input tokens**),
with better retention than either naive truncation.
