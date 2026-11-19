# ADR-0012: Complexity routing, semantic caching and degradation chains at the gateway
- Status: Accepted
- Date: 2026-11-19
- Projects: p11

## Context
Sending every request to the most capable model is the simplest policy and the most expensive. Many requests are simple or repeated paraphrases. Provider incidents and spend runaways must degrade service, not take it down or drain the budget.

## Decision
A gateway scores prompt complexity (entropy, length, lexical diversity, reasoning and code signals) and routes to Haiku / Sonnet / Opus tiers. A semantic cache with tenant isolation, TTL and LRU returns prior answers for near-duplicate prompts, with guards against false hits when numbers or negations differ. Per-tenant budgets and spend-rate circuit breakers act as kill-switches. Each tier has a fallback chain with per-backend circuit breakers.

## Consequences
40-60%+ cost reduction versus always-Opus on the replayed workload, with cache false-hit rate tracked as a first-class metric. Misrouting a hard prompt to a small model is the main quality risk, mitigated by conservative thresholds.

## Alternatives Considered
Always use the best model (simple, expensive). Exact-match caching only (low hit rate). Provider-side routing only (no tenant budgets or cross-provider fallback).
