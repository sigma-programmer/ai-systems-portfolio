# ADR-0008: Three memory tiers with importance-weighted recall and TTL eviction
- Status: Accepted
- Date: 2026-11-19
- Projects: p07, p01

## Context
Stuffing whole transcripts into context does not scale across sessions; a single vector store of every utterance recalls noise and stale facts ("I live in Paris" after the user moved).

## Decision
Working memory is a token-bounded scratchpad; overflow is compressed into episodic memory (session summaries); a consolidation pass distils semantic facts with supersession (newer contradicting facts replace older ones). Recall scores items by embedding similarity × importance × recency decay, across tiers. Each tier has its own TTL/decay. Everything persists in SQLite so a fresh process recalls prior sessions.

## Consequences
Bounded storage and context cost per user, recall dominated by salient and current facts. Importance weights are heuristic and tuned on a scripted benchmark rather than learned.

## Alternatives Considered
Full-transcript retrieval only (noisy, unbounded). Pure summarization chains (lose specific facts). Fact extraction only (loses narrative context).
