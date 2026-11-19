# ADR-0003: Fuse sparse and dense with RRF, then rerank a bounded candidate pool
- Status: Accepted
- Date: 2026-11-19
- Projects: p02, p04

## Context
BM25 excels at exact identifiers, rare terms and codes; dense retrieval excels at paraphrase and synonymy. Their raw scores live on incomparable scales. Cross-encoder reranking is the most accurate stage but cost scales linearly with candidates.

## Decision
Run BM25 and dense search in parallel, apply metadata filters before scoring, fuse with Reciprocal Rank Fusion (k=60) by default (weighted normalized fusion available), and rerank only the top-N fused candidates. N is a tuning knob exposed in the latency/accuracy sweep, which reports a Pareto frontier rather than a single "best" config.

## Consequences
Hybrid recall exceeds either retriever alone on mixed query sets; reranking cost is bounded and predictable. RRF ignores score magnitudes, so a retriever that is very confident gets no extra say — acceptable given its robustness to scale drift.

## Alternatives Considered
Dense-only (misses exact-match queries). Learned fusion weights per query type (needs labeled traffic). Reranking the full corpus (accurate, unaffordable).
