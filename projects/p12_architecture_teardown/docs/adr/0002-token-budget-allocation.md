# ADR-0002: Priority-tiered token budgeting with redistribution
- Status: Accepted
- Date: 2026-11-19
- Projects: p01

## Context
Context windows are large but finite, and quality degrades well before the hard limit (lost-in-the-middle, distraction by low-value retrieval). Naive concatenate-then-truncate drops whatever happens to come last — often the user's query or the newest turns.

## Decision
The Context Assembler reserves output tokens first, then satisfies hard-required blocks (system prompt, current query), then allocates the remainder across sources by priority weight within per-source min/max bounds, redistributing unused budget to sources that can use it. Each source declares a compression strategy (truncate, drop-oldest, top-k, summarize). The final render is verified against the window and the invariant is property-tested on randomized inputs.

## Consequences
Overflow is impossible by construction rather than by luck, and every assembly emits a report of what was included, compressed or dropped — making context decisions observable and debuggable. Token counts use a conservative estimator, so a small fraction of the window is left unused.

## Alternatives Considered
Fixed per-source quotas (wastes budget when a source is small). Pure recency truncation (drops instructions). Letting the API reject oversize requests and retrying (latency and cost spikes).
