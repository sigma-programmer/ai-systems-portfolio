# ADR-0005: Ground every claim and abstain below a calibrated confidence
- Status: Accepted
- Date: 2026-11-19
- Projects: p04

## Context
The costliest RAG failure is a fluent, confident, wrong answer. Users trust citations, so an answer citing a passage that doesn't support it is worse than no answer.

## Decision
Agentic RAG answers sentence-by-sentence with `[doc_id]` citations; a verifier checks each claim against its cited passage (entity, number and negation consistency). Unsupported claims are repaired or dropped; if the grounding ratio and retrieval evidence combine to a confidence below threshold, the system abstains and states what information is missing. Query rewriting and multi-step retrieval run before abstention is allowed.

## Consequences
Lower answer rate on hard questions, much lower hallucination rate. The abstention threshold is an explicit product knob traded off on the benchmark's answerable/unanswerable split.

## Alternatives Considered
Prompting the model to "only use the context" (insufficient on its own). Post-hoc LLM self-critique without citation alignment (cannot localize the unsupported claim).
