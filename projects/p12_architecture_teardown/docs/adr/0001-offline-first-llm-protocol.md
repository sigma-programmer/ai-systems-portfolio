# ADR-0001: Offline-first LLM protocol with deterministic defaults
- Status: Accepted
- Date: 2026-11-19
- Projects: all

## Context
Every subsystem calls an LLM somewhere (extraction, grading, judging, rewriting). If tests and benchmarks hit a live API they become slow, flaky, expensive and non-reproducible, and CI cannot gate on them. Mocking with canned strings hides whether the surrounding algorithm actually works.

## Decision
All projects depend only on a tiny `LLM` protocol (`complete(prompt, system, max_tokens, json_mode) -> LLMResponse`) from the shared `aikit` package. Each LLM-backed step ships two implementations: a genuine algorithmic default (pattern-based extraction, extractive summarization, heuristic judges) and an LLM-backed variant. `get_llm()` returns `ClaudeLLM` (claude-opus-5-5, adaptive thinking, server-side refusal fallback) only when credentials exist and `AIKIT_OFFLINE` is unset.

## Consequences
The entire portfolio runs, tests and benchmarks offline in seconds with fixed seeds, so benchmark fingerprints are reproducible. Offline quality is a floor, not a ceiling: swapping in Claude improves extraction/judging without touching orchestration code. Cost: two implementations per LLM step to maintain.

## Alternatives Considered
Recording/replaying real API responses (VCR-style) — brittle when prompts change. Using a local open-weights model — heavy dependency, still non-deterministic. Pure `MockLLM` canned replies — proves plumbing only, not algorithms.
