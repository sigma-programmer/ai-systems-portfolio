# P03 — GraphRAG Knowledge Pipeline

Turns unstructured documents into a typed knowledge graph, then answers
**multi-hop** questions by traversing it ("Who is the CEO of the company that
acquired the company that acquired Quillon Systems?") and **corpus-wide**
questions by map-reducing over community summaries ("Summarize the major
acquisitions in this industry.").

## Problem

Vector RAG retrieves passages that *look like* the question. A multi-hop
question names the start of the chain ("Quillon Systems"), so the most similar
chunks describe Quillon and its former CEO. The facts needed for hops 2 and 3
sit in documents that share almost no words with the question. A graph
stores those facts as edges. Answering then means following the chain from
the named entity, with every edge citing the chunk it came from.

## Architecture

```
 documents
    │
    ▼
 SentenceChunker ── sentence-aligned, token-budgeted chunks (ids "d04#0")
    │
    ▼
 Extractor (pluggable)
   ├─ HeuristicExtractor  capitalised-span NER + gazetteer/head-noun typing,
   │                      coreference ("the company", "She"), 30 relation
   │                      patterns (X acquired Y, X was founded by A and B,
   │                      X is CEO of Y, X is based in Y ...), alias patterns
   │                      "Nimbus Freight (NF)", co-occurrence fallback edges
   └─ LLMExtractor        JSON entities/relations via aikit.LLM; label
                          normalisation (FOUNDED_BY -> FOUNDED, reversed);
                          falls back to heuristics on malformed output
    │
    ▼
 EntityResolver ── union-find over normalised keys: suffix/article stripping
    │              ("Zentrix Labs Inc."), surname -> full name ("Okafor"),
    │              company stem ("Zentrix"), explicit aliases ("NF"), fuzzy
    │              typo match ("Quilon"), type-compatibility guard
    ▼
 KnowledgeGraph (networkx MultiDiGraph)
    │   nodes: name, type, aliases, mention_count, source_chunks
    │   edges keyed by relation: weight, source_chunks, evidence sentences
    │   type-signature check (PERSON ACQUIRED PERSON -> RELATED_TO)
    ▼
 Louvain (seeded, every level) ──► Community hierarchy L0 ⊂ L1
    │                                 + summaries (extractive | LLM)
    ▼
 ┌───────────────────────────────┐   ┌──────────────────────────────────────┐
 │ LocalSearch                   │   │ GlobalSearch                         │
 │ question -> logical form      │   │ MAP: score each community's facts/   │
 │  subj[CEO_OF](subj[ACQUIRED]( │   │   evidence vs question (embedding +  │
 │   Quillon Systems))           │   │   stem overlap + relation cue + anchor)│
 │ -> typed traversal, best path │   │   or LLM JSON points                 │
 │ -> answer + path + citations  │   │ REDUCE: relative cutoff, dedupe,     │
 │ fallback: k-hop neighbourhood │   │   top-k notes (or LLM synthesis)     │
 └───────────────────────────────┘   └──────────────────────────────────────┘
          ▲ GraphRAG.query() routes between them ▲
```

## Key design decisions and tradeoffs

- **The question becomes a logical form before any search.** A small
  recursive parser turns relative clauses ("the company that acquired X",
  "the company founded by X", "the company whose CEO is X", "the parent
  company of X") into nested `Hop` expressions. Running the expression is
  exact typed traversal. The parser only covers the patterns it knows, so
  anything else falls back to a scored k-hop neighbourhood search. With an
  LLM, the answer is phrased from the paths and cited passages.
- **Provenance is part of the data model.** Every node and edge carries
  `source_chunks`, and every edge keeps its evidence sentences. Answers
  return `supporting_chunks` and a readable path such as
  `Zentrix Labs -[ACQUIRED]-> Quillon Systems ; Alice Moreau -[CEO_OF]-> Zentrix Labs`.
- **Weights count documents, not mentions.** Re-extracting the same sentence
  (for example from overlapping chunks) never adds weight. When a node has
  several paths, the best one is chosen by its weakest edge (min weight).
- **Co-occurrence edges are kept but quarantined.** `RELATED_TO` edges at
  confidence 0.5 help community detection and the fallback search. Typed
  traversal ignores them, and community reports list typed facts first.
- **Resolution is conservative.** Fuzzy merges need similarity ≥ 0.9, names
  of at least 6 characters and compatible types, which keeps "Erik
  Halvorsen", "Ingrid Halvorsen" and "Halvorsen Group" apart. A surname
  or company stem merges only when it matches exactly one known entity.
- **Communities are hierarchical and deterministic.** `louvain_partitions`
  with a fixed seed gives every level, and parent/child links are recorded.
  Each community "owns" the edges whose subject it contains, so a boundary
  fact appears in exactly one report.
- **Everything works offline.** Extraction is pattern-based, summaries are
  extractive, and map/reduce uses heuristic scoring. Each of these steps can
  be swapped for an LLM through `aikit.LLM`.

## Honest caveats

The bundled corpus (26 fictional documents) and the heuristic extractor and
parser were developed together. The 100% extraction and local-search numbers
are therefore an upper bound for in-distribution text, not a claim about
open-domain prose. The point the benchmark *does* make holds regardless: with
the same chunks and the same answer-typing reader, vector RAG gets multi-hop
questions wrong because the bridging facts are never retrieved. On real data,
use `LLMExtractor` and keep the heuristics as a fallback.

## How to run

```bash
AIKIT_OFFLINE=1 .venv/bin/python projects/p03_graphrag/demo.py
.venv/bin/python -m pytest projects/p03_graphrag
cd projects/p03_graphrag && PYTHONPATH=../../common ../../.venv/bin/python -m graphrag.bench
```

```python
from graphrag import GraphRAG, DOCUMENTS
rag = GraphRAG().index(DOCUMENTS)            # or GraphRAG(extractor=LLMExtractor(llm), llm=llm)
ans = rag.local_search("Who is the CEO of the company that acquired Quillon Systems?")
print(ans.answer, ans.explanation, ans.supporting_chunks)
print(rag.global_search("Summarize the major acquisitions in this industry.").answer)
rag.save("kg.json")                          # nodes, edges, provenance, communities
```

## Benchmark (measured, `graphrag.bench.benchmark()`)

Corpus: 26 documents and 26 chunks (1,411 tokens). It has 26 QA items: 4
single-hop, 16 two-hop and 6 three-hop. Graph: 38 nodes and 80 edges (54
typed, 26 co-occurrence), with 10 level-0 and 7 level-1 communities.

| system | overall | multi-hop | 1-hop | 2-hop | 3-hop | gold-doc recall | ctx tokens | p50 latency |
|---|---|---|---|---|---|---|---|---|
| vector RAG top-3 | 0.269 | 0.136 | 1.00 | 0.12 | 0.17 | 0.577 | 176.0 | 0.27 ms |
| vector RAG top-5 | 0.269 | 0.136 | 1.00 | 0.12 | 0.17 | 0.692 | 288.5 | 0.35 ms |
| **GraphRAG local** | **1.000** | **1.000** | 1.00 | 1.00 | 1.00 | **1.000** | **127.7** | **0.05 ms** |

- **Retrieval depth:** going from top-3 to top-5 vector retrieval puts more
  gold documents in context (0.58 to 0.69) but doesn't fix a single answer.
  The extra chunks are mostly distractors that mention the question's own
  entity.
- **Context size:** GraphRAG cites fewer tokens (128 vs 289) because it only
  returns the chunks on the winning path.
- **Extraction vs. gold triples:** precision 1.00 and recall 1.00 (54/54).
- **Indexing time:** about 13 ms end to end (chunk+extract 5.6, resolve 4.0,
  graph 0.6, communities 2.7).
- **Global search:** p50 is 10 ms over 10 level-0 community reports
  (1,224 summary tokens), with an average of 2.7 points per answer.
- **LLM-mode indexing estimate:** about 7.4k input and 2.1k output tokens,
  roughly $0.09 at $5/$25 per million tokens. A global query costs 10 map
  calls plus 1 reduce call.
