# ADR-0004: Knowledge graph with provenance for multi-hop questions
- Status: Accepted
- Date: 2026-11-19
- Projects: p03

## Context
Questions like "Who is the CEO of the company that acquired X?" require joining facts that live in different documents. Vector RAG retrieves chunks similar to the question, but the bridging entity never appears in the question, so the second hop is usually missed.

## Decision
Convert unstructured text into a networkx knowledge graph (entities + typed relations), resolving aliases into canonical nodes and storing source-chunk provenance on every node and edge. Local search anchors on question entities and traverses typed paths; global search map-reduces over community summaries produced by modularity-based community detection.

## Consequences
Multi-hop accuracy improves substantially over naive vector RAG on the bundled benchmark, and every answer carries an explainable path with citations. Extraction quality bounds graph quality; the offline pattern extractor has limited recall versus an LLM extractor.

## Alternatives Considered
Iterative vector retrieval with query decomposition (works, but more LLM calls and no global view). A graph database (Neo4j) — operational overhead unjustified at this scale.
